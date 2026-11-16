"""Correctness and same-device timing against torch SDPA on BW1000."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn_bw import attention, build


def prepare_timing(fn, mode, inner, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                result = fn()

        def run():
            graph.replay()
            return result  # Keep the captured output allocation alive.
    else:
        def run():
            for _ in range(inner):
                fn()
    return run


def measure_pair(functions, mode, inner, warmup, repeats):
    runs = [prepare_timing(fn, mode, inner, warmup) for fn in functions]
    samples = [[], []]
    # Alternate measurement order to reduce clock/thermal order bias.
    for repeat in range(repeats):
        for i in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            runs[i]()
            end.record()
            end.synchronize()
            samples[i].append(begin.elapsed_time(end) / inner)
    return [statistics.median(x) for x in samples]


def selected_sdpa_ops(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
    torch.cuda.synchronize()
    return sorted({event.key for event in prof.key_averages()
                   if "scaled_dot_product" in event.key and event.key != "aten::scaled_dot_product_attention"})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--kv-length", type=int, help="fixed M for cross-attention; otherwise M=N")
    parser.add_argument("--backend", choices=["auto", "flash", "efficient", "math"], default="auto")
    parser.add_argument("--timing", choices=["graph", "event"], default="graph")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", type=Path, help="save device, settings and measurements")
    args = parser.parse_args()
    positive = [args.batch, args.heads, *args.lengths, args.warmup, args.inner, args.repeats]
    if min(positive) < 1 or (args.kv_length is not None and args.kv_length < 1):
        parser.error("shapes and timing counts must be positive")
    if torch.version.hip is None or not torch.cuda.is_available():
        parser.error("run this script with DTK PyTorch on BW1000")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "").split(":")[0]
    if arch != "gfx936":
        parser.error(f"requires gfx936, found {arch!r}")
    lib = build(head_dim=args.head_dim)
    scale = args.head_dim ** -0.5
    metadata = {"torch": torch.__version__, "hip": torch.version.hip,
                "device": props.name, "arch": arch, "dtype": "float16",
                "compute_units": getattr(props, "multi_processor_count", None),
                "shared_memory_per_block": getattr(props, "shared_memory_per_block", None),
                "layout": "BHND", "head_dim": args.head_dim, "causal": False,
                "dropout": 0.0, "scale": scale,
                "exp2": lib.fa_bw_exp2_mode().decode(),
                "q_tile": lib.fa_bw_q_tile(),
                "kv_tile": lib.fa_bw_kv_tile(),
                "lds_bytes": 8 * lib.fa_bw_kv_tile() * args.head_dim,
                "prefetch": lib.fa_bw_prefetch_mode().decode(),
                "v_load": lib.fa_bw_v_load_mode().decode(),
                "full_kv_specialization": bool(lib.fa_bw_full_kv_enabled()),
                "async_v": bool(lib.fa_bw_async_v_enabled()),
                "async_k": bool(lib.fa_bw_async_k_enabled()),
                "conditional_rescale": bool(lib.fa_bw_conditional_rescale_enabled()),
                "wave_rescale": bool(lib.fa_bw_wave_rescale_enabled()),
                "wave_rescale_scope": "full_kv",
                "defer_denom": bool(lib.fa_bw_defer_denom_enabled()),
                "defer_denom_scope": "full_kv",
                "reduce4": bool(lib.fa_bw_reduce4_enabled()),
                "delay_scale": bool(lib.fa_bw_delay_scale_enabled()),
                "delay_scale_scope": "full_kv_positive_scale",
                "HIPCC": os.environ.get("HIPCC"),
                "FA_BW_CXXFLAGS": os.environ.get("FA_BW_CXXFLAGS", ""),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    print(json.dumps(metadata, indent=2))
    print("Latency: median GPU-event time per call; graph mode excludes Python dispatch overhead.")
    print(f"{'B':>3} {'H':>3} {'N':>6} {'M':>6} {'BW us':>10} {'SDPA us':>10} {'speedup':>9} {'BW TF/s':>9} {'max err':>10}")
    rows = []
    backends = {"flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
    context = nullcontext() if args.backend == "auto" else sdpa_kernel(backends=[backends[args.backend]])
    with context:
        for n in args.lengths:
            m = n if args.kv_length is None else args.kv_length
            torch.manual_seed(42)
            q = torch.randn(args.batch, args.heads, n, args.head_dim, device="cuda", dtype=torch.float16)
            k = torch.randn(args.batch, args.heads, m, args.head_dim, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            out = torch.empty_like(q)
            custom = lambda: attention(q, k, v, out=out)
            sdpa = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            actual, expected = custom(), sdpa()
            torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
            # Independent FP32 reference on sampled query rows avoids an N*M
            # allocation at long sequence lengths. Compare ALL rows with SDPA.
            idx = torch.linspace(0, n - 1, min(17, n), device="cuda").long()
            reference = (torch.softmax(q[:, :, idx].float() @ k.float().transpose(-1, -2) * scale, -1)
                         @ v.float()).half()
            torch.testing.assert_close(actual[:, :, idx], reference, atol=2e-3, rtol=2e-3)
            max_error = (actual.float() - expected.float()).abs().max().item()
            ops = selected_sdpa_ops(sdpa)
            print("SDPA operators:", ", ".join(ops) or "not identified by CPU profiler", flush=True)
            custom_ms, sdpa_ms = measure_pair((custom, sdpa), args.timing, args.inner, args.warmup, args.repeats)
            flops = 4 * args.batch * args.heads * n * m * args.head_dim
            row = {"B": args.batch, "H": args.heads, "N": n, "M": m, "D": args.head_dim,
                   "custom_us": custom_ms * 1000, "sdpa_us": sdpa_ms * 1000,
                   "speedup": sdpa_ms / custom_ms, "custom_tflops": flops / (custom_ms * 1e9),
                   "sdpa_tflops": flops / (sdpa_ms * 1e9),
                   "max_abs_error": max_error, "sdpa_operators": ops}
            rows.append(row)
            print(f"{args.batch:3} {args.heads:3} {n:6} {m:6} {row['custom_us']:10.2f} "
                  f"{row['sdpa_us']:10.2f} {row['speedup']:8.3f}x {row['custom_tflops']:9.2f} {max_error:10.3g}", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    main()
