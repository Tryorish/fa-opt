"""One fixed attention workload for hipprof; eager launches, no graph/profiler nesting."""

import argparse
from contextlib import nullcontext
import json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from flash_attn_bw import attention, build


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["custom", "sdpa"], default="custom")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if min(args.length, args.batch, args.heads, args.repeats) < 1 or args.warmup < 0:
        parser.error("shapes/repeats must be positive and warmup must be nonnegative")
    if torch.version.hip is None or not torch.cuda.is_available():
        parser.error("requires DTK HIP PyTorch on BW1000")
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "").split(":")[0]
    if arch != "gfx936":
        parser.error(f"requires gfx936, got {arch!r}")

    metadata = {**vars(args), "torch": torch.__version__, "hip": torch.version.hip,
                "device": props.name, "arch": arch, "dtype": "float16",
                "layout": "BHND", "head_dim": args.head_dim, "scale": args.head_dim ** -0.5,
                "causal": False, "graph": False,
                "attention_calls": args.warmup + args.repeats}
    if args.impl == "custom":
        lib = build(head_dim=args.head_dim)
        metadata.update(q_tile=lib.fa_bw_q_tile(), kv_tile=lib.fa_bw_kv_tile(),
                        exp2=lib.fa_bw_exp2_mode().decode(),
                        v_load=lib.fa_bw_v_load_mode().decode(),
                        full_kv_specialization=bool(lib.fa_bw_full_kv_enabled()),
                        async_v=bool(lib.fa_bw_async_v_enabled()),
                        async_k=bool(lib.fa_bw_async_k_enabled()),
                        conditional_rescale=bool(lib.fa_bw_conditional_rescale_enabled()),
                        wave_rescale=bool(lib.fa_bw_wave_rescale_enabled()),
                        wave_rescale_scope="full_kv",
                        defer_denom=bool(lib.fa_bw_defer_denom_enabled()),
                        defer_denom_scope="full_kv",
                        reduce4=bool(lib.fa_bw_reduce4_enabled()),
                        delay_scale=bool(lib.fa_bw_delay_scale_enabled()),
                        delay_scale_scope="full_kv_positive_scale",
                        prefetch=lib.fa_bw_prefetch_mode().decode())
    else:
        metadata["sdpa_backend"] = "FLASH_ATTENTION (forced)"
    print(json.dumps(metadata, indent=2), flush=True)

    # Generate on CPU so GPU random/conversion kernels don't pollute PMC data.
    generator = torch.Generator(device="cpu").manual_seed(42)
    shape = (args.batch, args.heads, args.length, args.head_dim)
    q, k, v = [torch.randn(shape, generator=generator).half().to("cuda") for _ in range(3)]
    out = torch.empty_like(q)
    context = nullcontext() if args.impl == "custom" else sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    fn = (lambda: attention(q, k, v, out=out)) if args.impl == "custom" else (
        lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False))
    torch.cuda.synchronize()
    with context:
        for _ in range(args.warmup):
            result = fn()
        torch.cuda.synchronize()
        print("Warmup complete; collecting repeated attention calls.", flush=True)
        for _ in range(args.repeats):
            result = fn()
        torch.cuda.synchronize()
    # Keep result alive through synchronization without launching validation kernels.
    print(f"Completed {args.warmup + args.repeats} calls; output shape={tuple(result.shape)}", flush=True)


if __name__ == "__main__":
    main()
