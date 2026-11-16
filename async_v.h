#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "shared_layout.h"

#ifndef FA_BW_ASYNC_V
#define FA_BW_ASYNC_V 0
#endif
static_assert(FA_BW_ASYNC_V == 0 || FA_BW_ASYNC_V == 1, "FA_BW_ASYNC_V must be 0 or 1");

// Complete tiles only: every 16B request is in bounds. The gfx936 path in
// tilelang-hygon's hcu/copy.h uses this buffer_load_dwordx4 ... lds form.
template <int Keys>
__device__ __forceinline__ void async_v_to_lds(const _Float16* source, _Float16* dest) {
    static_assert(Keys == 32 || Keys == 64 || Keys == 128, "V tile must contain 32, 64 or 128 keys");
#if defined(__HIP_DEVICE_COMPILE__) && FA_BW_ASYNC_V
    using resource4 = uint32_t __attribute__((ext_vector_type(4)));
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    const resource4 resource = {
        static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(uint32_t(address))),
        static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(uint32_t(address >> 32))),
        Keys * HEAD_DIM * sizeof(_Float16), 0x00020000u};  // gfx936 raw buffer format.
    const int tid = threadIdx.x;
    #pragma unroll
    for (int chunk = 0; chunk < Keys * HEAD_DIM / (256 * 8); ++chunk) {
        // m0 is the wave destination base; hardware adds lane*16 bytes.
        const auto wave_ptr = dest + (tid / 64) * 512 + chunk * 2048;
        const uint32_t lds_base = __builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(wave_ptr)));
        const uint32_t offset = async_v_source(tid, chunk) * sizeof(_Float16);
        // m0 is reserved by the compiler. Preserve it inside one asm block;
        // early-clobber keeps the saved value separate from every input SGPR.
        uint32_t saved_m0;
        asm volatile("s_mov_b32 %0, m0\n\t"
                     "s_mov_b32 m0, %1\n\t"
                     "buffer_load_dwordx4 %2, %3, 0 offen lds\n\t"
                     "s_mov_b32 m0, %0\n\t"
                     : "=&s"(saved_m0) : "s"(lds_base), "v"(offset), "s"(resource) : "memory");
    }
#endif
}

__device__ __forceinline__ void wait_async_v() {
#if defined(__HIP_DEVICE_COMPILE__) && FA_BW_ASYNC_V
    // Complete each wave's writes before the caller's block-wide barrier.
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" : : : "memory");
#endif
}
