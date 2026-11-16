"""CPU layout proof plus real-kernel tests (the latter require BW1000)."""

import unittest
import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def layout_attention(q, k, v, keys=32, *, defer_denom=False, reduce4=False,
                     wave_rescale=False, delay_scale=False, scale=0.125):
    """Model wave64 register ownership; deliberately independent of HIP code."""
    lane = torch.arange(64)
    row, group = lane % 16, lane // 16
    reg = torch.arange(4)
    # Physical column -> original key, as proposed by the user.
    perm = torch.tensor([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])
    out = torch.zeros(16, v.shape[-1])
    maximum = torch.full((16,), -torch.inf)
    denominator = torch.zeros(64 if defer_denom else 16)
    delay_scale = delay_scale and scale > 0 and len(k) % keys == 0
    for start in range(0, k.shape[0], keys):
        kt = F.pad(k[start:start + keys], (0, 0, 0, max(0, start + keys - len(k))))
        vt = F.pad(v[start:start + keys], (0, 0, 0, max(0, start + keys - len(v))))
        physical_keys = torch.cat([perm + j * 16 for j in range(keys // 16)])
        scores = q @ kt[physical_keys].T
        if not delay_scale:
            scores *= scale
        scores[:, start + physical_keys >= len(k)] = -torch.inf
        # MMAC's C registers: [lane, 16-key tile, register].
        c_cols = group[:, None] + 4 * reg
        regs = torch.stack([scores[row[:, None], c_cols + j * 16] for j in range(keys // 16)], 1)
        # Reduce the four lanes for each row (xor 16, xor 32 on hardware).
        if reduce4:
            maxima = regs[:, 0].clone()
            for j in range(1, keys // 16):
                maxima = torch.maximum(maxima, regs[:, j])
            local_max = torch.maximum(torch.maximum(maxima[:, 0], maxima[:, 1]),
                                      torch.maximum(maxima[:, 2], maxima[:, 3]))
        else:
            local_max = regs.amax((1, 2))
        tile_max = local_max.reshape(4, 16).amax(0)
        if delay_scale:
            tile_max *= scale
        new_max = torch.maximum(maximum, tile_max)
        probs = torch.exp((regs * scale if delay_scale else regs) - new_max[row, None, None])
        if reduce4:
            sums = probs[:, 0].clone()
            for j in range(1, keys // 16):
                sums += probs[:, j]
            tile_sum = (sums[:, 0] + sums[:, 1]) + (sums[:, 2] + sums[:, 3])
        else:
            tile_sum = probs.sum((1, 2))
        if not defer_denom:
            tile_sum = tile_sum.reshape(4, 16).sum(0)
        # Reinterpret C registers directly as A, with NO lane exchange.
        p = torch.zeros(16, keys)
        a_cols = 4 * group[:, None] + reg
        for j in range(keys // 16):
            p[row[:, None], a_cols + j * 16] = probs[:, j].half().float()
        if not wave_rescale or torch.any(new_max > maximum):
            correction = torch.exp(maximum - new_max)
            out = out * correction[:, None]
            denominator = denominator * (correction[row] if defer_denom else correction)
        out = out + p @ vt
        denominator = denominator + tile_sum
        maximum = new_max
    if defer_denom:
        denominator = denominator.reshape(4, 16).sum(0)
    return out / denominator[:, None]


class LayoutProof(unittest.TestCase):
    def test_delayed_scale_with_signed_logits(self):
        torch.manual_seed(103)
        q = torch.zeros(16, 64)
        q[:, 0] = torch.linspace(-2, 2, 16)
        k = torch.zeros(257, 64)
        k[:, 0] = torch.linspace(-32, 32, 257)
        v = torch.randn(257, 64).half().float()
        for scale in (0.01, 0.125, 0.7, 2.0, 0.0, -0.25):
            for m in (128, 256, 257):
                expected = torch.softmax(q @ k[:m].T * scale, -1) @ v[:m]
                for keys in (32, 64, 128):
                    with self.subTest(scale=scale, m=m, keys=keys):
                        actual = layout_attention(q, k[:m], v[:m], keys, scale=scale,
                                                  defer_denom=True, delay_scale=True)
                        torch.testing.assert_close(actual, expected, atol=6e-4, rtol=2e-3)

    def test_wave_rescale_with_one_changing_row(self):
        torch.manual_seed(91)
        k = torch.zeros(257, 64)
        k[:, 0] = torch.linspace(0, 32, 257)
        v = torch.randn(257, 64).half().float()
        for slopes in (torch.zeros(16), torch.ones(16),
                       torch.tensor([-1.] * 15 + [1.])):
            q = torch.zeros(16, 64)
            q[:, 0] = slopes
            for m in (128, 256, 257):
                expected = torch.softmax(q @ k[:m].T * 0.125, -1) @ v[:m]
                for keys in (32, 64, 128):
                    with self.subTest(slopes=slopes.tolist(), m=m, keys=keys):
                        actual = layout_attention(q, k[:m], v[:m], keys,
                                                  defer_denom=True, wave_rescale=True)
                        torch.testing.assert_close(actual, expected, atol=6e-4, rtol=2e-3)

    def test_four_way_reductions_match_dense_attention(self):
        torch.manual_seed(84)
        q = torch.zeros(16, 64)
        q[:, 0] = torch.linspace(-2, 2, 16)
        # Unequal register weights plus increasing/decreasing tile maxima.
        key = torch.arange(1025)
        k = torch.zeros(1025, 64)
        k[:, 0] = torch.tensor([0., 32., -48., 16.])[key % 4] + key / 64
        v = torch.randn(1025, 64).half().float()
        for m in (1, 64, 128, 1024, 1025):
            expected = torch.softmax(q @ k[:m].T * 0.125, -1) @ v[:m]
            for keys in (32, 64, 128):
                with self.subTest(m=m, keys=keys):
                    actual = layout_attention(q, k[:m], v[:m], keys,
                                              defer_denom=True, reduce4=True)
                    torch.testing.assert_close(actual, expected, atol=6e-4, rtol=2e-3)

    def test_deferred_denominator_with_unequal_lane_sums(self):
        # Different groups must contribute different amounts. A missing final
        # reduction or an accidental factor of four must not pass this case.
        torch.manual_seed(73)
        q = torch.zeros(16, 64)
        q[:, 0] = torch.linspace(-2, 2, 16)
        k = torch.zeros(257, 64)
        key = torch.arange(257)
        k[:, 0] = torch.tensor([0., 48., -32., 16.])[(key % 16) // 4] + key / 32
        v = torch.randn(257, 64).half().float()
        for m in (64, 256, 257):
            expected = torch.softmax(q @ k[:m].T * 0.125, -1) @ v[:m]
            for keys in (32, 64, 128):
                with self.subTest(m=m, keys=keys):
                    actual = layout_attention(q, k[:m], v[:m], keys, defer_denom=True)
                    torch.testing.assert_close(actual, expected, atol=6e-4, rtol=2e-3)

    @unittest.skipUnless(shutil.which("c++"), "requires a host C++ compiler")
    def test_production_lds_indices_preserve_vectors_and_spread_banks(self):
        source = Path(__file__).with_name("test_lds_layout.cpp")
        with tempfile.TemporaryDirectory() as tmp:
            executable = str(Path(tmp) / "lds_layout")
            subprocess.run(["c++", "-std=c++17", str(source), "-o", executable], check=True)
            result = subprocess.run([executable], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_register_reuse_matches_dense_attention(self):
        # Catches a wrong permutation direction, missing tile offsets, wrong
        # lane ownership, a permuted V, and incorrect online rescaling.
        torch.manual_seed(17)
        for m in (1, 17, 32, 33, 63, 64, 65, 97, 127, 128, 129, 256, 257):
            with self.subTest(m=m):
                q, k, v = (torch.randn(n, 64).half().float() for n in (16, m, m))
                expected = torch.softmax(q @ k.T * 0.125, -1) @ v
                for keys in (32, 64, 128):
                    with self.subTest(keys=keys):
                        torch.testing.assert_close(layout_attention(q, k, v, keys), expected,
                                                   atol=6e-4, rtol=2e-3)


def on_bw1000():
    return (torch.version.hip is not None and torch.cuda.is_available()
            and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0] == "gfx936")


def check_probe_sources(root=Path(__file__).resolve().parent):
    # D64-only probe allocations are unsafe with the D128 copy helpers.
    # Detect a partial directory update before compiling or running a probe.
    marker = "#define FA_BW_PROBE_LAYOUT_VERSION 2"
    for name in ("test_matrix_v.hip", "test_async_k.hip"):
        source = root / name
        if not source.is_file() or marker not in source.read_text().splitlines():
            raise RuntimeError(f"Outdated or missing probe: {source}. "
                               "Copy the current test_matrix_v.hip and test_async_k.hip "
                               "together with test_flash_attn.py; D128 requires layout version 2.")


class InputValidation(unittest.TestCase):
    def test_contiguous_view_must_still_be_vector_aligned(self):
        from flash_attn_bw import attention
        # contiguous() alone does not guarantee alignment of an offset view.
        q = torch.zeros(65, dtype=torch.float16)[1:].view(1, 1, 1, 64)
        kv = torch.zeros_like(q)
        self.assertTrue(q.is_contiguous())
        with self.assertRaisesRegex(ValueError, "align"):
            attention(q, kv, kv)


@unittest.skipUnless(on_bw1000(), "requires BW1000/gfx936 and DTK PyTorch")
class KernelTests(unittest.TestCase):
    head_dim = 64
    copy_tiles = (32, 64, 128)

    @classmethod
    def setUpClass(cls):
        from flash_attn_bw import build
        check_probe_sources()
        build(head_dim=cls.head_dim)

    def check_case(self, b, h, n, m, magnitude=1.0, scale=None, zero_q=False):
        from flash_attn_bw import attention
        torch.manual_seed(42)
        q = torch.randn(b, h, n, self.head_dim, device="cuda", dtype=torch.float16) * magnitude
        k = torch.randn(b, h, m, self.head_dim, device="cuda", dtype=torch.float16) * magnitude
        v = torch.randn_like(k)
        if zero_q:
            q.zero_()
        actual = attention(q, k, v, scale=scale)
        s = self.head_dim ** -0.5 if scale is None else scale
        expected = (torch.softmax(q.float() @ k.float().transpose(-1, -2) * s, -1)
                    @ v.float()).half()
        self.assertTrue(torch.isfinite(actual).all(), "custom kernel returned non-finite values")
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        # DTK torch 2.7.1 Flash SDPA produced NaNs for scale<=0
        # on BW1000. These valid kernel cases still check against the explicit
        # FP32 reference AND SDPA Math; positive scales keep normal dispatch.
        context = sdpa_kernel(backends=[SDPBackend.MATH]) if s <= 0 else nullcontext()
        with context:
            sdpa = F.scaled_dot_product_attention(q, k, v, scale=s)
        self.assertTrue(torch.isfinite(sdpa).all(), f"SDPA reference is non-finite for scale={s}")
        torch.testing.assert_close(actual, sdpa, atol=2e-3, rtol=2e-3)

    def test_tails_batches_and_cross_attention(self):
        for shape in ((1, 1, 1, 1), (1, 2, 17, 19), (2, 3, 65, 33),
                      (1, 1, 64, 32), (2, 2, 63, 63), (1, 2, 64, 64),
                      (1, 3, 65, 65), (1, 1, 65, 31), (1, 1, 65, 32), (1, 2, 97, 129),
                      (1, 2, 127, 127), (2, 2, 129, 128), (1, 2, 129, 256),
                      (1, 2, 128, 257), (1, 2, 513, 1024)):
            with self.subTest(shape=shape):
                self.check_case(*shape)

    def test_scaling_and_softmax_stability(self):
        for m in (64, 97, 128, 129):  # Full C64/C128 tiles and masked tails.
            for kwargs in ({"magnitude": 12.0}, {"scale": 0.0}, {"scale": -0.25},
                           {"magnitude": 12.0, "scale": -0.25}, {"zero_q": True}):
                with self.subTest(m=m, kwargs=kwargs):
                    self.check_case(1, 2, 67, m, **kwargs)

    def test_positive_scales_with_shifted_logits(self):
        from flash_attn_bw import attention
        torch.manual_seed(103)
        q = torch.zeros(1, 2, 257, self.head_dim, device="cuda", dtype=torch.float16)
        q[..., 0] = torch.linspace(-2, 2, 257, device="cuda")
        k = torch.zeros(1, 2, 4097, self.head_dim, device="cuda", dtype=torch.float16)
        key = torch.linspace(-24, 24, 4097, device="cuda")
        # Both signs of the raw maximum and a large shared logit offset.
        k[0, 0, :, 0] = key + 128
        k[0, 1, :, 0] = key - 128
        v = torch.randn_like(k)
        for scale in (0.01, 0.125, 0.7, 2.0):
            for m in (128, 4096, 4097):
                with self.subTest(scale=scale, m=m):
                    km, vm = k[:, :, :m].contiguous(), v[:, :, :m].contiguous()
                    expected = (torch.softmax(q.float() @ km.float().transpose(-1, -2) * scale, -1)
                                @ vm.float()).half()
                    actual = attention(q, km, vm, scale=scale)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    def test_matrix_v_fragment_layout(self):
        from flash_attn_bw import build
        if build(head_dim=self.head_dim).fa_bw_v_load_mode().decode() != "matrix":
            self.skipTest("matrix V is disabled; comparison script also tests the enabled path")
        compiler = os.environ.get("HIPCC") or shutil.which("aicc") or shutil.which("hipcc")
        source = Path(__file__).with_name("test_matrix_v.hip")
        with tempfile.TemporaryDirectory() as tmp:
            executable = str(Path(tmp) / "matrix_v_probe")
            subprocess.run([compiler, "-x", "hip", "-O3", "-std=c++17", "--offload-arch=gfx936",
                            f"-DFA_BW_HEAD_DIM={self.head_dim}",
                            "-DFA_BW_MATRIX_V=1", f"-DFA_BW_TEST_KEYS={build(head_dim=self.head_dim).fa_bw_kv_tile()}",
                            str(source), "-o", executable], check=True)
            subprocess.run([executable], check=True)

    def test_query_tiles_and_guarded_output(self):
        from flash_attn_bw import attention
        torch.manual_seed(26)
        # Q192 must cover its third part and ceil-divide across 192/384 rows.
        # Check both full-KV and masked paths, including output staging reuse.
        shapes = ((n, m) for n in (127, 128, 129, 191, 192, 193,
                                   255, 256, 257, 383, 384, 385) for m in (97, 128))
        for n, m in shapes:
            with self.subTest(n=n, m=m):
                q = torch.randn(2, 2, n, self.head_dim, device="cuda", dtype=torch.float16)
                k = torch.randn(2, 2, m, self.head_dim, device="cuda", dtype=torch.float16)
                v = torch.randn_like(k)
                # Different query groups need independent online max/sum/O.
                gain = 0.5 + (torch.arange(n, device="cuda") // 16) % 4
                q *= gain[:, None]
                storage = torch.full((q.numel() + 128,), 123.0, device="cuda", dtype=q.dtype)
                out = storage[64:-64].view_as(q)
                out.fill_(float("nan"))  # Missing rows cannot silently pass.
                actual = attention(q, k, v, out=out)
                self.assertEqual(actual.data_ptr(), out.data_ptr())
                expected = (torch.softmax(q.float() @ k.float().transpose(-1, -2) * self.head_dim ** -0.5, -1)
                            @ v.float()).half()
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
                self.assertTrue((storage[:64] == 123).all())
                self.assertTrue((storage[-64:] == 123).all())

    def test_async_v_copy_and_fragment_layout(self):
        from flash_attn_bw import build
        if not build(head_dim=self.head_dim).fa_bw_async_v_enabled():
            self.skipTest("async V is disabled")
        compiler = os.environ.get("HIPCC") or shutil.which("aicc") or shutil.which("hipcc")
        source = Path(__file__).with_name("test_matrix_v.hip")
        with tempfile.TemporaryDirectory() as tmp:
            for keys in self.copy_tiles:
                executable = str(Path(tmp) / f"async_v_{keys}")
                subprocess.run([compiler, "-x", "hip", "-O3", "-std=c++17", "--offload-arch=gfx936",
                                f"-DFA_BW_HEAD_DIM={self.head_dim}",
                                "-DFA_BW_MATRIX_V=1", "-DFA_BW_ASYNC_V=1", f"-DFA_BW_TEST_KEYS={keys}",
                                str(source), "-o", executable], check=True)
                subprocess.run([executable], check=True)

    def test_async_k_copy_and_guards(self):
        from flash_attn_bw import build
        if not build(head_dim=self.head_dim).fa_bw_async_k_enabled():
            self.skipTest("async K is disabled")
        compiler = os.environ.get("HIPCC") or shutil.which("aicc") or shutil.which("hipcc")
        source = Path(__file__).with_name("test_async_k.hip")
        with tempfile.TemporaryDirectory() as tmp:
            for keys in self.copy_tiles:
                executable = str(Path(tmp) / f"async_k_{keys}")
                subprocess.run([compiler, "-x", "hip", "-O3", "-std=c++17", "--offload-arch=gfx936",
                                f"-DFA_BW_HEAD_DIM={self.head_dim}",
                                "-Werror=inline-asm", "-DFA_BW_ASYNC_V=1", "-DFA_BW_ASYNC_K=1",
                                f"-DFA_BW_TEST_KEYS={keys}", str(source), "-o", executable], check=True)
                subprocess.run([executable], check=True)

    def test_online_softmax_many_small_max_updates(self):
        from flash_attn_bw import attention
        torch.manual_seed(18)
        q = torch.zeros(1, 2, 257, self.head_dim, device="cuda", dtype=torch.float16)
        k = torch.zeros(1, 2, 4097, self.head_dim, device="cuda", dtype=torch.float16)
        slopes = torch.linspace(0.5, 1.5, 257, device="cuda")
        slopes[1::2].neg_()
        q[..., 0] = slopes
        k[..., 0] = torch.linspace(-64, 64, 4097, device="cuda")
        v = torch.randn_like(k)
        # Even queries update their maximum every tile; odd queries keep
        # the first maximum. Each wave therefore mixes both rescale paths.
        # Reversing keys swaps their roles without changing the answer.
        for m in (4096, 4097):  # Exercise the pipelined full path and tail fallback.
            km, vm = k[:, :, :m].contiguous(), v[:, :, :m].contiguous()
            reference = (torch.softmax(q.float() @ km.float().transpose(-1, -2) * self.head_dim ** -0.5, -1)
                         @ vm.float()).half()
            for reverse in (False, True):
                with self.subTest(m=m, reverse=reverse):
                    kk, vv = (km.flip(-2), vm.flip(-2)) if reverse else (km, vm)
                    actual = attention(q, kk, vv)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)

    def test_unequal_lane_denominators(self):
        from flash_attn_bw import attention
        q = torch.zeros(1, 2, 129, self.head_dim, device="cuda", dtype=torch.float16)
        q[..., 0] = torch.linspace(-2, 2, 129, device="cuda")
        k = torch.zeros(1, 2, 4097, self.head_dim, device="cuda", dtype=torch.float16)
        key = torch.arange(4097, device="cuda")
        group = (key % 16) // 4
        bias = torch.tensor([0., 48., -32., 16.], device="cuda")
        k[0, 0, :, 0] = bias[group] + key / 512
        k[0, 1, :, 0] = bias[(group + 1) % 4] + key / 512
        # Each output group exposes its probability mass. Use equal V across
        # heads so rotating the dominant key group must rotate the answer.
        v = (group[:, None] == torch.arange(self.head_dim, device="cuda") % 4).half()
        v = v[None, None].expand_as(k).contiguous()
        for m in (64, 128, 4096, 4097):
            with self.subTest(m=m):
                km, vm = k[:, :, :m].contiguous(), v[:, :, :m].contiguous()
                expected = (torch.softmax(q.float() @ km.float().transpose(-1, -2) * self.head_dim ** -0.5, -1)
                            @ vm.float()).half()
                actual = attention(q, km, vm)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    def test_wave_rescale_with_one_changing_row(self):
        from flash_attn_bw import attention
        torch.manual_seed(91)
        q = torch.zeros(1, 2, 257, self.head_dim, device="cuda", dtype=torch.float16)
        k = torch.zeros(1, 2, 4097, self.head_dim, device="cuda", dtype=torch.float16)
        k[..., 0] = torch.linspace(0, 32, 4097, device="cuda")
        v = torch.randn_like(k)
        for mode in ("none", "all", "last_row"):
            q.zero_()
            if mode == "all":
                q[..., 0] = 1
            elif mode == "last_row":
                # Lane0's row never updates after tile0; row15 always does.
                # Reading only lane0's predicate must not replace a wave vote.
                q[..., 0] = -1
                q[..., 15::16, 0] = 1
            for m in (128, 4096, 4097):
                with self.subTest(mode=mode, m=m):
                    km, vm = k[:, :, :m].contiguous(), v[:, :, :m].contiguous()
                    expected = (torch.softmax(q.float() @ km.float().transpose(-1, -2) * self.head_dim ** -0.5, -1)
                                @ vm.float()).half()
                    actual = attention(q, km, vm)
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    def test_current_stream(self):
        # Input production, kernel launch and result consumption must all use
        # the caller's stream. A default-stream launch races this test.
        with torch.cuda.stream(torch.cuda.Stream()):
            for m in (64, 97, 128, 129):
                self.check_case(1, 2, 65, m)

    def test_reject_invalid_inputs(self):
        from flash_attn_bw import attention
        q = torch.zeros(1, 2, 64, self.head_dim, device="cuda", dtype=torch.float16)
        cases = ((q.float(), q, q), (q.transpose(-1, -2), q, q),
                 (q[..., :32], q, q), (q, q[:, :1], q[:, :1]),
                 (q, q[..., :0, :], q[..., :0, :]))
        for args in cases:
            with self.subTest(shapes=[x.shape for x in args]):
                with self.assertRaises((ValueError, TypeError)):
                    attention(*args)


if __name__ == "__main__":
    unittest.main()
