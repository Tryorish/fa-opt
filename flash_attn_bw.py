"""Build and call the standalone BW1000 HIP kernel. No TileLang dependency."""

import ctypes
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import torch

_ROOT = Path(__file__).resolve().parent
_LIBS = {}


def build(verbose=True, *, head_dim=64):
    """Compile/cache each head dimension separately with the active DTK compiler."""
    if head_dim not in (64, 128):
        raise ValueError("head_dim must be 64 or 128")
    if head_dim in _LIBS:
        return _LIBS[head_dim]
    source = _ROOT / "flash_attn_bw.hip"
    name = "libflash_attn_bw.so" if head_dim == 64 else "libflash_attn_bw_d128.so"
    library = _ROOT / "build" / name
    compiler = os.environ.get("HIPCC") or shutil.which("aicc") or shutil.which("hipcc")
    if not compiler:
        raise RuntimeError("DTK compiler not found: put hipcc/aicc on PATH or set HIPCC=/path/to/hipcc")
    flags = ["-x", "hip", "-O3", "-std=c++17", "--offload-arch=gfx936", "-fPIC", "-shared"]
    flags += shlex.split(os.environ.get("FA_BW_CXXFLAGS", ""))
    # The tensor dimension selects the binary; an environment macro must not
    # silently give a D128 tensor to a D64 kernel (or vice versa).
    flags += ["-UFA_BW_HEAD_DIM", f"-DFA_BW_HEAD_DIM={head_dim}"]
    inputs = (source, _ROOT / "shared_layout.h", _ROOT / "matrix_v.h", _ROOT / "async_v.h", _ROOT / "async_k.h")
    signature = "\n".join([str(compiler), *flags,
                           *(f"{p.name}:{p.stat().st_mtime_ns}" for p in inputs)])
    stamp = library.with_suffix(".stamp")
    if not library.exists() or not stamp.exists() or stamp.read_text() != signature:
        library.parent.mkdir(exist_ok=True)
        # Atomic publication prevents another process from loading a partial .so.
        with tempfile.TemporaryDirectory(dir=library.parent) as tmp:
            output = Path(tmp) / library.name
            command = [compiler, *flags, str(source), "-o", str(output)]
            if verbose:
                print(shlex.join(command), flush=True)
            subprocess.run(command, check=True)
            output.replace(library)
        stamp.write_text(signature)
    lib = ctypes.CDLL(str(library))
    lib.fa_bw_forward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    lib.fa_bw_forward.restype = ctypes.c_int
    lib.fa_bw_error.argtypes = [ctypes.c_int]
    lib.fa_bw_error.restype = ctypes.c_char_p
    lib.fa_bw_exp2_mode.argtypes = []
    lib.fa_bw_exp2_mode.restype = ctypes.c_char_p
    lib.fa_bw_kv_tile.argtypes = []
    lib.fa_bw_kv_tile.restype = ctypes.c_int
    lib.fa_bw_q_tile.argtypes = []
    lib.fa_bw_q_tile.restype = ctypes.c_int
    lib.fa_bw_head_dim.argtypes = []
    lib.fa_bw_head_dim.restype = ctypes.c_int
    lib.fa_bw_full_kv_enabled.argtypes = []
    lib.fa_bw_full_kv_enabled.restype = ctypes.c_int
    lib.fa_bw_async_v_enabled.argtypes = []
    lib.fa_bw_async_v_enabled.restype = ctypes.c_int
    lib.fa_bw_async_k_enabled.argtypes = []
    lib.fa_bw_async_k_enabled.restype = ctypes.c_int
    lib.fa_bw_conditional_rescale_enabled.argtypes = []
    lib.fa_bw_conditional_rescale_enabled.restype = ctypes.c_int
    lib.fa_bw_wave_rescale_enabled.argtypes = []
    lib.fa_bw_wave_rescale_enabled.restype = ctypes.c_int
    lib.fa_bw_defer_denom_enabled.argtypes = []
    lib.fa_bw_defer_denom_enabled.restype = ctypes.c_int
    lib.fa_bw_reduce4_enabled.argtypes = []
    lib.fa_bw_reduce4_enabled.restype = ctypes.c_int
    lib.fa_bw_delay_scale_enabled.argtypes = []
    lib.fa_bw_delay_scale_enabled.restype = ctypes.c_int
    lib.fa_bw_v_load_mode.argtypes = []
    lib.fa_bw_v_load_mode.restype = ctypes.c_char_p
    lib.fa_bw_prefetch_mode.argtypes = []
    lib.fa_bw_prefetch_mode.restype = ctypes.c_char_p
    if lib.fa_bw_head_dim() != head_dim:
        raise RuntimeError("compiled head dimension does not match requested dimension")
    _LIBS[head_dim] = lib
    return lib


def attention(q, k, v, *, scale=None, out=None):
    """FP16 contiguous [B,H,N,D] x [B,H,M,D], D=64/128; noncausal forward."""
    tensors = (q, k, v)
    if any(x.ndim != 4 or x.shape[-1] not in (64, 128) for x in tensors):
        raise ValueError("Q/K/V must have shape [B,H,sequence,D] with D=64 or 128")
    if any(x.shape[-1] != q.shape[-1] for x in tensors):
        raise ValueError("Q/K/V must have the same head dimension")
    if any(x.dtype != torch.float16 for x in tensors):
        raise TypeError("Q/K/V must be FP16")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("Q/K/V must be contiguous in [B,H,sequence,D] order")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("Q/K/V must share B/H, and K/V must have identical shapes")
    if any(x.numel() == 0 for x in tensors):
        raise ValueError("Empty tensors are not supported")
    if any(x.data_ptr() % 16 for x in tensors):
        raise ValueError("Q/K/V data pointers must be 16-byte aligned")
    if not q.is_cuda or any(x.device != q.device for x in tensors):
        raise ValueError("Q/K/V must be on the same BW1000 device")
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise ValueError("Forward only: use torch.no_grad() for tensors requiring gradients")
    if torch.version.hip is None:
        raise RuntimeError("Use DTK's HIP PyTorch on BW1000")
    arch = getattr(torch.cuda.get_device_properties(q.device), "gcnArchName", "").split(":")[0]
    if arch != "gfx936":
        raise RuntimeError(f"This kernel requires gfx936, got {arch!r}")
    heads, n, m = q.shape[0] * q.shape[1], q.shape[2], k.shape[2]
    # Leave room for the final start + KV_TILE and loop increment at C128.
    if heads > 65535 or max(n, m) > 2**31 - 129:
        raise ValueError("B*H must be <=65535 and sequence lengths must fit the kernel's int32 indices")
    s = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(s) or abs(s) > 1e30:
        raise ValueError("scale must be finite with abs(scale) <= 1e30")
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device or not out.is_contiguous():
        raise ValueError("out must have Q's shape, dtype, device and be contiguous")
    if out.data_ptr() % 16:
        raise ValueError("out must be 16-byte aligned")
    if torch.is_grad_enabled() and out.requires_grad:
        raise ValueError("Forward only: out must not require gradients")
    if any(out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in tensors):
        raise ValueError("out must not share storage with Q/K/V")
    lib = build(head_dim=q.shape[-1])
    with torch.cuda.device(q.device):
        stream = torch.cuda.current_stream(q.device)
        status = lib.fa_bw_forward(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                                   heads, n, m, s, stream.cuda_stream)
        if status:
            raise RuntimeError(lib.fa_bw_error(status).decode())
        # Raw pointer calls bypass PyTorch's dispatcher: retain stream ownership.
        for x in (*tensors, out):
            x.record_stream(stream)
    return out


if __name__ == "__main__":
    build()
