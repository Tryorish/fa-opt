#pragma once

#ifndef FA_BW_HEAD_DIM
#define FA_BW_HEAD_DIM 64
#endif
static_assert(FA_BW_HEAD_DIM == 64 || FA_BW_HEAD_DIM == 128, "Head dimension must be 64 or 128");
constexpr int HEAD_DIM = FA_BW_HEAD_DIM;

#if defined(__HIPCC__) || defined(__HIP__)
#define FA_BW_HD __host__ __device__ __forceinline__
#else
#define FA_BW_HD inline
#endif

// Four-half groups remain contiguous and eight-byte aligned.
template <int Stride>
FA_BW_HD constexpr int lds(int row, int col) {
    return row * Stride + (col ^ ((row % (Stride / 4)) * 4));
}

template <int Stride>
FA_BW_HD constexpr int lds_v(int row, int col) {
    // V producer threads visit output columns eight apart. Fold those high
    // bits into the bank selection, for both 32-key and 64-key V tiles.
    const int phase = (row ^ (row / 8)) % (Stride / 4);
    return row * Stride + (col ^ (phase * 4));
}

// Row-major V[key,channel], preserving each 16-byte vector. Odd key rows
// exchange adjacent 32-channel panels to spread matrix-read bank requests.
FA_BW_HD constexpr int lds_v_matrix(int row, int col) {
    return row * HEAD_DIM + (col ^ ((row & 1) * 32));
}

// Direct-to-LDS writes are linear in lane id. Apply the inverse swizzle
// (the same XOR) to GLOBAL source columns instead of to LDS destinations.
FA_BW_HD constexpr int async_v_source(int tid, int chunk) {
    const int i = (tid + chunk * 256) * 8;
    return lds_v_matrix(i / HEAD_DIM, i % HEAD_DIM);
}

// K's XOR preserves four-half vectors, so direct copies use 8B per lane.
FA_BW_HD constexpr int async_k_source(int tid, int chunk) {
    const int i = (tid + chunk * 256) * 4;
    return lds<HEAD_DIM>(i / HEAD_DIM, i % HEAD_DIM);
}

#undef FA_BW_HD
