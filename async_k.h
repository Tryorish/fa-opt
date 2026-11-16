#pragma once
#include "async_v.h"

#ifndef FA_BW_ASYNC_K
#define FA_BW_ASYNC_K 0
#endif
static_assert(FA_BW_ASYNC_K == 0 || FA_BW_ASYNC_K == 1, "FA_BW_ASYNC_K must be 0 or 1");
static_assert(!FA_BW_ASYNC_K || FA_BW_ASYNC_V, "Async K requires async V (shared wait/pipeline)");

// Complete tiles only. K's existing swizzle preserves 8B, not 16B vectors.
// gfx936 supports buffer_load_dwordx2 ... lds (see hcu/copy.h).
template <int Keys>
__device__ __forceinline__ void async_k_to_lds(const _Float16* source, _Float16* dest) {
    static_assert(Keys == 32 || Keys == 64 || Keys == 128, "K tile must contain 32, 64 or 128 keys");
#if defined(__HIP_DEVICE_COMPILE__) && FA_BW_ASYNC_K
    using resource4 = uint32_t __attribute__((ext_vector_type(4)));
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    const resource4 resource = {
        static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(uint32_t(address))),
        static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(uint32_t(address >> 32))),
        Keys * HEAD_DIM * sizeof(_Float16), 0x00020000u};  // gfx936 raw buffer format.
    const int tid = threadIdx.x;
    #pragma unroll
    for (int chunk = 0; chunk < Keys * HEAD_DIM / (256 * 4); ++chunk) {
        // m0 is the wave base; hardware adds lane*8 bytes. Offsets here are halves.
        const auto wave_ptr = dest + (tid / 64) * 256 + chunk * 1024;
        const uint32_t lds_base = __builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(wave_ptr)));
        const uint32_t offset = async_k_source(tid, chunk) * sizeof(_Float16);
        uint32_t saved_m0;
        asm volatile("s_mov_b32 %0, m0\n\t"
                     "s_mov_b32 m0, %1\n\t"
                     "buffer_load_dwordx2 %2, %3, 0 offen lds\n\t"
                     "s_mov_b32 m0, %0\n\t"
                     : "=&s"(saved_m0) : "s"(lds_base), "v"(offset), "s"(resource) : "memory");
    }
#endif
}
