#pragma once
#include <hip/hip_runtime.h>
#include "shared_layout.h"

using half4 = _Float16 __attribute__((ext_vector_type(4)));
using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float4v = float __attribute__((ext_vector_type(4)));

#ifndef FA_BW_MATRIX_V
#define FA_BW_MATRIX_V 1
#endif
static_assert(FA_BW_MATRIX_V == 0 || FA_BW_MATRIX_V == 1,
              "FA_BW_MATRIX_V must be 0 or 1");

// A 16-key x 32-channel panel becomes two MMAC B fragments per lane:
// result[4*f+r] = V[key_tile*16 + 4*(lane/16)+r, panel*32 + 16*f+lane%16].
// Request lanes differ from destination lanes: each requests 8 adjacent halves.
// See tilelang_ds_read_format.hpp: ds_read_format_tensor_b_linear.
__device__ __forceinline__ half8 load_v_matrix(_Float16* sv, int key_tile, int panel) {
#if defined(__HIP_DEVICE_COMPILE__) && FA_BW_MATRIX_V
    const int lane = threadIdx.x % 64;
    auto* ptr = sv + lds_v_matrix(key_tile * 16 + lane / 4, panel * 32 + (lane % 4) * 8);
    // The return type is a 128-bit container, not four numerical FP32 values.
    return __builtin_bit_cast(half8, __builtin_hcu_ds_read_m32x16_f16(ptr, 0));
#else
    return half8{};  // Unused in the host pass and the scalar-V configuration.
#endif
}
