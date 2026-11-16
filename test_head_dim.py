"""D128 regression suite; CPU layout checks run without a BW1000."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

import test_flash_attn as base


class HeadDimLayout(unittest.TestCase):
    def test_d128_register_reuse_and_online_softmax(self):
        torch.manual_seed(128)
        for m in (1, 32, 64, 97, 256, 257):
            q, k, v = (torch.randn(n, 128).half().float() for n in (16, m, m))
            scale = 128 ** -0.5
            expected = torch.softmax(q @ k.T * scale, -1) @ v
            for keys in (32, 64):
                for optimized in (False, True):
                    with self.subTest(m=m, keys=keys, optimized=optimized):
                        actual = base.layout_attention(q, k, v, keys, scale=scale,
                                                       defer_denom=optimized, delay_scale=optimized)
                        torch.testing.assert_close(actual, expected, atol=6e-4, rtol=2e-3)

    @unittest.skipUnless(shutil.which("c++"), "requires a host C++ compiler")
    def test_d128_copies_and_matrix_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / "layout")
            source = Path(__file__).with_name("test_head_dim_layout.cpp")
            subprocess.run(["c++", "-std=c++17", "-DFA_BW_HEAD_DIM=128", str(source), "-o", exe], check=True)
            result = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_d128_alignment_reaches_vector_validation(self):
        from flash_attn_bw import attention
        q = torch.zeros(129, dtype=torch.float16)[1:].view(1, 1, 1, 128)
        kv = torch.zeros_like(q)
        with self.assertRaisesRegex(ValueError, "align"):
            attention(q, kv, kv)

    def test_mismatched_head_dimensions(self):
        from flash_attn_bw import attention
        q = torch.zeros(1, 1, 1, 128, dtype=torch.float16)
        k = torch.zeros(1, 1, 1, 64, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "same head dimension"):
            attention(q, k, k)


class KernelTests128(base.KernelTests):
    head_dim = 128
    copy_tiles = (32, 64)

    def test_second_half_channels_and_default_scale(self):
        from flash_attn_bw import attention
        torch.manual_seed(128)
        for n, m in ((65, 32), (129, 64), (129, 97), (257, 4096)):
            with self.subTest(n=n, m=m):
                q = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.float16)
                k = torch.randn(1, 2, m, 128, device="cuda", dtype=torch.float16)
                q[..., :64] = 0
                k[..., :64] = 0
                v = torch.randn_like(k)
                v[..., 64:] += 2  # Distinguish upper output panels from lower ones.
                expected = (torch.softmax(q.float() @ k.float().transpose(-1, -2) / 128**0.5, -1)
                            @ v.float()).half()
                actual = attention(q, k, v)
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    def test_d64_and_d128_in_same_process(self):
        from flash_attn_bw import attention, build
        for d in (128, 64, 128):
            with self.subTest(d=d):
                q, k, v = [torch.randn(1, 2, 128, d, device="cuda", dtype=torch.float16) for _ in range(3)]
                self.assertEqual(build(head_dim=d).fa_bw_head_dim(), d)
                expected = (torch.softmax(q.float() @ k.float().transpose(-1, -2) / d**0.5, -1)
                            @ v.float()).half()
                torch.testing.assert_close(attention(q, k, v), expected, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
