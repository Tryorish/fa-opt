// BW1000 (gfx936), FP16 [B,H,N,D], D=64/128, noncausal forward attention.
// Four wave64s/block; 64/128/192/256 query rows, 32/64/128 keys per iteration.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "matrix_v.h"
#include "async_k.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx936__)
#error "This kernel requires BW1000 / gfx936 (wave64, MMAC without LIT)."
#endif

#ifndef FA_BW_NATIVE_EXP2
#define FA_BW_NATIVE_EXP2 1
#endif
#if FA_BW_NATIVE_EXP2 != 0 && FA_BW_NATIVE_EXP2 != 1
#error "FA_BW_NATIVE_EXP2 must be 0 or 1."
#endif

#ifndef FA_BW_LATE_PREFETCH
#define FA_BW_LATE_PREFETCH 0
#endif
static_assert(FA_BW_LATE_PREFETCH == 0 || FA_BW_LATE_PREFETCH == 1,
              "FA_BW_LATE_PREFETCH must be 0 or 1");

#ifndef FA_BW_KV_TILE
#define FA_BW_KV_TILE (FA_BW_HEAD_DIM == 128 ? 32 : 64)
#endif
static_assert(FA_BW_KV_TILE == 32 || FA_BW_KV_TILE == 64 || FA_BW_KV_TILE == 128,
              "KV tile must be 32, 64 or 128");
static_assert(FA_BW_KV_TILE != 128 || FA_BW_MATRIX_V, "C128 requires matrix V");
static_assert(8 * FA_BW_KV_TILE * HEAD_DIM <= 65536, "K/V double buffers must fit 64 KiB LDS");
constexpr int KV_TILE = FA_BW_KV_TILE;
constexpr int KV_VECS = KV_TILE * HEAD_DIM / (256 * 8);
constexpr int SCORE_TILES = KV_TILE / 16;
constexpr int HEAD_TILES = HEAD_DIM / 16;

#ifndef FA_BW_Q_TILE
#define FA_BW_Q_TILE (FA_BW_HEAD_DIM == 128 ? 64 : 128)
#endif
static_assert(FA_BW_Q_TILE == 64 || FA_BW_Q_TILE == 128 ||
              FA_BW_Q_TILE == 192 || FA_BW_Q_TILE == 256,
              "Q tile must be 64, 128, 192 or 256");
constexpr int Q_TILE = FA_BW_Q_TILE;
constexpr int Q_PARTS = Q_TILE / 64;  // Each wave owns 16 rows per part.

#ifndef FA_BW_FULL_KV
#define FA_BW_FULL_KV 1
#endif
static_assert(FA_BW_FULL_KV == 0 || FA_BW_FULL_KV == 1,
              "FA_BW_FULL_KV must be 0 or 1");
static_assert(!FA_BW_ASYNC_V || (FA_BW_MATRIX_V && FA_BW_FULL_KV && !FA_BW_LATE_PREFETCH),
              "Async V requires matrix V, full-KV specialization and early prefetch");

#ifndef FA_BW_CONDITIONAL_RESCALE
#define FA_BW_CONDITIONAL_RESCALE 0
#endif
static_assert(FA_BW_CONDITIONAL_RESCALE == 0 || FA_BW_CONDITIONAL_RESCALE == 1,
              "FA_BW_CONDITIONAL_RESCALE must be 0 or 1");

#ifndef FA_BW_WAVE_RESCALE
#define FA_BW_WAVE_RESCALE 0
#endif
static_assert(FA_BW_WAVE_RESCALE == 0 || FA_BW_WAVE_RESCALE == 1,
              "FA_BW_WAVE_RESCALE must be 0 or 1");
static_assert(!(FA_BW_WAVE_RESCALE && FA_BW_CONDITIONAL_RESCALE),
              "Select either wave-uniform or per-row conditional rescaling");

#ifndef FA_BW_DEFER_DENOM
#define FA_BW_DEFER_DENOM 0
#endif
static_assert(FA_BW_DEFER_DENOM == 0 || FA_BW_DEFER_DENOM == 1,
              "FA_BW_DEFER_DENOM must be 0 or 1");

#ifndef FA_BW_REDUCE4
#define FA_BW_REDUCE4 0
#endif
static_assert(FA_BW_REDUCE4 == 0 || FA_BW_REDUCE4 == 1,
              "FA_BW_REDUCE4 must be 0 or 1");

#ifndef FA_BW_DELAY_SCALE
#define FA_BW_DELAY_SCALE 0
#endif
static_assert(FA_BW_DELAY_SCALE == 0 || FA_BW_DELAY_SCALE == 1,
              "FA_BW_DELAY_SCALE must be 0 or 1");

// Max subtraction gives nonpositive inputs in exact arithmetic (also -inf
// for masked keys); FMA may leave a positive rounding residual at the maximum.
// Select the hardware instruction locally; do not enable global fast-math.
__device__ __forceinline__ float softmax_exp2(float x) {
#if defined(__HIP_DEVICE_COMPILE__) && FA_BW_NATIVE_EXP2
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

__device__ __forceinline__ float4v mmac(half4 a, half4 b, float4v c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_hcu_mmac_f32_16x16x16_f16(a, b, c);
#else
    return c;  // HIP's host compilation pass does not execute device code.
#endif
}

template <bool FullKV>
__device__ __forceinline__ half8 load_kv(const _Float16* p, int start, int m, int chunk) {
    const int i = (threadIdx.x + chunk * 256) * 8;
    return FullKV || start + i / HEAD_DIM < m
        ? *reinterpret_cast<const half8*>(p + int64_t(start) * HEAD_DIM + i)
        : half8{};
}

template <bool FullKV, bool DirectK, bool DirectV>
__device__ __forceinline__ void prefetch_kv(
    const _Float16* k, const _Float16* v, int start, int m,
    half8* next_k, half8* next_v) {
    #pragma unroll
    for (int chunk = 0; chunk < KV_VECS; ++chunk) {
        if constexpr (!DirectK)
            next_k[chunk] = load_kv<FullKV>(k, start, m, chunk);
        if constexpr (!DirectV)
            next_v[chunk] = load_kv<FullKV>(v, start, m, chunk);
    }
}

template <bool DirectK, bool DirectV>
__device__ __forceinline__ void stage_kv(
    _Float16* sk, _Float16* sv, half8 k, half8 v, int chunk) {
    const int i = (threadIdx.x + chunk * 256) * 8;
    const int row = i / HEAD_DIM, col = i % HEAD_DIM;
    if constexpr (!DirectK) {
        *reinterpret_cast<half4*>(sk + lds<HEAD_DIM>(row, col)) =
            __builtin_shufflevector(k, k, 0, 1, 2, 3);
        *reinterpret_cast<half4*>(sk + lds<HEAD_DIM>(row, col + 4)) =
            __builtin_shufflevector(k, k, 4, 5, 6, 7);
    }
    if constexpr (DirectV) {
        // V was already issued directly into its LDS buffer.
    } else if constexpr (FA_BW_MATRIX_V) {
        // Keep V row-major: the matrix read produces MMAC B layout.
        *reinterpret_cast<half8*>(sv + lds_v_matrix(row, col)) = v;
    } else {
        #pragma unroll
        for (int r = 0; r < 8; ++r)
            sv[lds_v<KV_TILE>(col + r, row)] = v[r];
    }
}

template <bool FullKV>
__global__ __launch_bounds__(256)
void flash_attn_v2_r64c32d64_kernel(
    const _Float16* __restrict__ q, const _Float16* __restrict__ k,
    const _Float16* __restrict__ v, _Float16* __restrict__ o,
    int n, int m, float scale) {
    constexpr bool DirectK = FullKV && FA_BW_ASYNC_K;
    constexpr bool DirectV = FullKV && FA_BW_ASYNC_V;
    // Avoid the extra register pressure measured in the masked-tail kernel.
    constexpr bool DeferDenom = FullKV && FA_BW_DEFER_DENOM;
    constexpr bool Reduce4 = FullKV && FA_BW_REDUCE4;
    constexpr bool WaveRescale = FullKV && FA_BW_WAVE_RESCALE;
    // The launcher sends nonpositive scales to the original masked path.
    constexpr bool DelayScale = FullKV && FA_BW_DELAY_SCALE;
    const int tid = threadIdx.x;
    const int wave = tid / 64, lane = tid % 64;
    const int row = lane % 16, group = lane / 16;
    const int64_t query_start = int64_t(blockIdx.x) * Q_TILE;
    const int64_t head = blockIdx.y;
    q += head * n * HEAD_DIM;
    k += head * m * HEAD_DIM;
    v += head * m * HEAD_DIM;
    o += head * n * HEAD_DIM;

    // Q/output staging reuse K/V storage: 8 * KV_TILE * HEAD_DIM bytes.
    union alignas(16) SharedStorage {
        _Float16 qo[64 * HEAD_DIM];
        struct { _Float16 k[2][KV_TILE * HEAD_DIM], v[2][KV_TILE * HEAD_DIM]; } kv;
    };
    __shared__ SharedStorage sm;

    // Stage 64 rows at a time: larger Q tiles do not increase LDS usage.
    half4 qr[Q_PARTS][HEAD_TILES];
    #pragma unroll
    for (int part = 0; part < Q_PARTS; ++part) {
        // 256 个线程，每个线程一次搬 4 个 half
        #pragma unroll
        for (int i = tid * 4; i < 64 * HEAD_DIM; i += 256 * 4) {
            const int r = i / HEAD_DIM, c = i % HEAD_DIM;
            const int64_t query = query_start + part * 64 + r;
            *reinterpret_cast<half4*>(sm.qo + lds<HEAD_DIM>(r, c)) = query < n
                ? *reinterpret_cast<const half4*>(q + query * HEAD_DIM + c)
                : half4{};
        }
        __syncthreads();
        #pragma unroll
        for (int d = 0; d < HEAD_TILES; ++d)
            qr[part][d] = *reinterpret_cast<half4*>(sm.qo + lds<HEAD_DIM>(wave * 16 + row, d * 16 + group * 4));
        __syncthreads();  // Finish reads before staging another part or K/V.
    }

    if constexpr (DirectK)
        async_k_to_lds<KV_TILE>(k, sm.kv.k[0]);
    if constexpr (DirectV)
        async_v_to_lds<KV_TILE>(v, sm.kv.v[0]);
    #pragma unroll
    for (int chunk = 0; chunk < KV_VECS; ++chunk) {
        half8 kr{}, vr{};
        if constexpr (!DirectK) kr = load_kv<FullKV>(k, 0, m, chunk);
        if constexpr (!DirectV) vr = load_kv<FullKV>(v, 0, m, chunk);
        stage_kv<DirectK, DirectV>(sm.kv.k[0], sm.kv.v[0], kr, vr, chunk);
    }
    if constexpr (DirectV) wait_async_v();
    __syncthreads();
    float4v acc[Q_PARTS][HEAD_TILES] = {};
    float maximum[Q_PARTS], denominator[Q_PARTS] = {};
    #pragma unroll
    for (int part = 0; part < Q_PARTS; ++part)
        maximum[part] = -INFINITY;
    const float log2_scale = scale * 1.4426950408889634f;

    for (int start = 0, iteration = 0; start < m; start += KV_TILE, ++iteration) {
        const int current = iteration & 1;
        const bool has_next = start + KV_TILE < m;
        // Previous iteration's barrier finished reads of this alternate buffer.
        if constexpr (DirectK)
            if (has_next)
                async_k_to_lds<KV_TILE>(k + int64_t(start + KV_TILE) * HEAD_DIM, sm.kv.k[current ^ 1]);
        if constexpr (DirectV)
            if (has_next)
                async_v_to_lds<KV_TILE>(v + int64_t(start + KV_TILE) * HEAD_DIM, sm.kv.v[current ^ 1]);
        // Register-prefetched tiles are staged after PV. The late variant
        // loads after softmax, shortening overlap with QK's live temporaries.
        half8 next_k[KV_VECS]{}, next_v[KV_VECS]{};
        if constexpr (!FA_BW_LATE_PREFETCH)
            if (has_next)
                prefetch_kv<FullKV, DirectK, DirectV>(k, v, start + KV_TILE, m, next_k, next_v);

        float4v score[Q_PARTS][SCORE_TILES] = {};
        #pragma unroll
        for (int j = 0; j < SCORE_TILES; ++j) {
            // Physical QK column j*16+row reads original key j*16+perm(row).
            const int key = j * 16 + (row % 4) * 4 + row / 4;
            #pragma unroll
            for (int d = 0; d < HEAD_TILES; ++d) {
                half4 kr = *reinterpret_cast<half4*>(sm.kv.k[current] + lds<HEAD_DIM>(key, d * 16 + group * 4));
                // Reuse this K fragment across independent query parts.
                #pragma unroll
                for (int part = 0; part < Q_PARTS; ++part)
                    score[part][j] = mmac(qr[part][d], kr, score[part][j]);
            }
        }

        half4 p[Q_PARTS][SCORE_TILES];
        #pragma unroll
        for (int part = 0; part < Q_PARTS; ++part) {
            float tile_max = -INFINITY;
            float maxima[4]{};
            #pragma unroll
            for (int j = 0; j < SCORE_TILES; ++j) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    // C's physical column group+4*r represents original key 4*group+r.
                    const int key = start + j * 16 + group * 4 + r;
                    if constexpr (!DelayScale)
                        score[part][j][r] = FullKV || key < m ? score[part][j][r] * log2_scale : -INFINITY;
                    if constexpr (Reduce4)
                        maxima[r] = j == 0 ? score[part][j][r] : fmaxf(maxima[r], score[part][j][r]);
                    else
                        tile_max = fmaxf(tile_max, score[part][j][r]);
                }
            }
            if constexpr (Reduce4)
                tile_max = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16, 64));
            tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 32, 64));
            // Positive scaling commutes with max. Scale one row maximum,
            // instead of each score just to discover that maximum.
            if constexpr (DelayScale)
                tile_max *= log2_scale;
            const float new_max = fmaxf(maximum[part], tile_max);

            float tile_sum = 0.0f;
            float sums[4]{};
            #pragma unroll
            for (int j = 0; j < SCORE_TILES; ++j) {
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const float shifted = DelayScale
                        ? fmaf(score[part][j][r], log2_scale, -new_max)
                        : score[part][j][r] - new_max;
                    const float probability = softmax_exp2(shifted);
                    // Four independent register-column chains, then a tree.
                    if constexpr (Reduce4)
                        sums[r] = j == 0 ? probability : sums[r] + probability;
                    else
                        tile_sum += probability;
                    p[part][j][r] = (_Float16)probability;
                }
            }
            if constexpr (Reduce4)
                tile_sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
            if constexpr (!DeferDenom) {
                tile_sum += __shfl_xor(tile_sum, 16, 64);
                tile_sum += __shfl_xor(tile_sum, 32, 64);
            }
            // If every row in this wave keeps its maximum, every correction
            // is exactly 1. Vote with all 64 lanes before branching; a single
            // changing row makes the WHOLE wave execute the original math.
            bool rescale = !FA_BW_CONDITIONAL_RESCALE || new_max > maximum[part];
            if constexpr (WaveRescale)
                rescale = __any(new_max > maximum[part]);
            // Shuffles and MMACs stay outside the optional rescale branch.
            if (rescale) {
                const float correction = softmax_exp2(maximum[part] - new_max);
                denominator[part] = denominator[part] * correction + tile_sum;
                #pragma unroll
                for (int d = 0; d < HEAD_TILES; ++d)
                    acc[part][d] *= correction;
            } else {
                denominator[part] += tile_sum;
            }
            maximum[part] = new_max;
        }

        if constexpr (FA_BW_LATE_PREFETCH) {
            // Consume P before the memory compiler barrier so next-tile loads
            // cannot be hoisted back above the softmax computation. This emits
            // no GPU instruction and is not a cross-thread synchronization.
            #pragma unroll
            for (int part = 0; part < Q_PARTS; ++part) {
                #pragma unroll
                for (int j = 0; j < SCORE_TILES; ++j)
                    asm volatile("" : : "v"(p[part][j]) : "memory");
            }
            if (has_next)
                prefetch_kv<FullKV, DirectK, DirectV>(k, v, start + KV_TILE, m, next_k, next_v);
        }

        // Both paths consume the same P fragments, with no P shuffles/LDS.
        if constexpr (FA_BW_MATRIX_V) {
            #pragma unroll
            for (int panel = 0; panel < HEAD_DIM / 32; ++panel) {
                #pragma unroll
                for (int j = 0; j < SCORE_TILES; ++j) {
                    const half8 vr = load_v_matrix(sm.kv.v[current], j, panel);
                    const half4 b[2] = {__builtin_shufflevector(vr, vr, 0, 1, 2, 3),
                                        __builtin_shufflevector(vr, vr, 4, 5, 6, 7)};
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        #pragma unroll
                        for (int part = 0; part < Q_PARTS; ++part)
                            acc[part][panel * 2 + f] = mmac(p[part][j], b[f], acc[part][panel * 2 + f]);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_TILES; ++d) {
                #pragma unroll
                for (int j = 0; j < SCORE_TILES; ++j) {
                    half4 vr = *reinterpret_cast<half4*>(sm.kv.v[current] + lds_v<KV_TILE>(d * 16 + row, j * 16 + group * 4));
                    #pragma unroll
                    for (int part = 0; part < Q_PARTS; ++part)
                        acc[part][d] = mmac(p[part][j], vr, acc[part][d]);
                }
            }
        }

        if (has_next) {
            #pragma unroll
            for (int chunk = 0; chunk < KV_VECS; ++chunk)
                stage_kv<DirectK, DirectV>(sm.kv.k[current ^ 1], sm.kv.v[current ^ 1], next_k[chunk], next_v[chunk], chunk);
            if constexpr (DirectV) wait_async_v();
        }
        __syncthreads();  // Finish current reads AND publish the next tile.
    }

    // MMAC C -> row-major output, reusing the same 64-row scratch per part.
    #pragma unroll
    for (int part = 0; part < Q_PARTS; ++part) {
        float sum = denominator[part];
        if constexpr (DeferDenom) {
            // The four lanes for this row share every maximum/correction.
            // Sum their independently rescaled FP32 denominators only once.
            sum += __shfl_xor(sum, 16, 64);
            sum += __shfl_xor(sum, 32, 64);
        }
        const float inv_sum = 1.0f / sum;
        #pragma unroll
        for (int d = 0; d < HEAD_TILES; ++d) {
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                sm.qo[lds<HEAD_DIM>(wave * 16 + row, d * 16 + group + r * 4)] = (_Float16)(acc[part][d][r] * inv_sum);
        }
        __syncthreads();
        #pragma unroll
        for (int i = tid * 4; i < 64 * HEAD_DIM; i += 256 * 4) {
            const int r = i / HEAD_DIM, c = i % HEAD_DIM;
            const int64_t query = query_start + part * 64 + r;
            if (query < n)
                *reinterpret_cast<half4*>(o + query * HEAD_DIM + c) =
                    *reinterpret_cast<half4*>(sm.qo + lds<HEAD_DIM>(r, c));
        }
        if (part + 1 < Q_PARTS)
            __syncthreads();  // Finish coalesced reads before overwriting scratch.
    }
}

template <bool FullKV>
int launch(const void* q, const void* k, const void* v, void* o,
           int heads, int n, int m, float scale, void* stream) {
    hipLaunchKernelGGL(flash_attn_v2_r64c32d64_kernel<FullKV>,
                      dim3((n - 1) / Q_TILE + 1, heads), dim3(256), 0,
                      reinterpret_cast<hipStream_t>(stream),
                      static_cast<const _Float16*>(q), static_cast<const _Float16*>(k),
                      static_cast<const _Float16*>(v), static_cast<_Float16*>(o), n, m, scale);
    return static_cast<int>(hipGetLastError());
}

// Small C ABI: complete KV tiles may omit bounds checks; tails always keep them.
extern "C" int fa_bw_forward(const void* q, const void* k, const void* v, void* o,
                             int heads, int n, int m, float scale, void* stream) {
    if constexpr (FA_BW_FULL_KV) {
        // Positive scaling preserves max order; zero/negative keep the original path.
        if (m % KV_TILE == 0 && (!FA_BW_DELAY_SCALE || scale > 0.0f))
            return launch<true>(q, k, v, o, heads, n, m, scale, stream);
    }
    return launch<false>(q, k, v, o, heads, n, m, scale, stream);
}

extern "C" const char* fa_bw_error(int status) {
    return hipGetErrorString(static_cast<hipError_t>(status));
}

extern "C" const char* fa_bw_exp2_mode() {
    return FA_BW_NATIVE_EXP2 ? "native" : "libm";
}

extern "C" int fa_bw_kv_tile() { return KV_TILE; }

extern "C" int fa_bw_q_tile() { return Q_TILE; }

extern "C" int fa_bw_full_kv_enabled() { return FA_BW_FULL_KV; }

extern "C" int fa_bw_async_v_enabled() { return FA_BW_ASYNC_V; }

extern "C" int fa_bw_async_k_enabled() { return FA_BW_ASYNC_K; }

extern "C" int fa_bw_conditional_rescale_enabled() { return FA_BW_CONDITIONAL_RESCALE; }

extern "C" int fa_bw_wave_rescale_enabled() { return FA_BW_WAVE_RESCALE; }

extern "C" int fa_bw_defer_denom_enabled() { return FA_BW_DEFER_DENOM; }

extern "C" int fa_bw_reduce4_enabled() { return FA_BW_REDUCE4; }

extern "C" int fa_bw_delay_scale_enabled() { return FA_BW_DELAY_SCALE; }

extern "C" const char* fa_bw_v_load_mode() {
    return FA_BW_MATRIX_V ? "matrix" : "transposed";
}

extern "C" const char* fa_bw_prefetch_mode() {
    return FA_BW_LATE_PREFETCH ? "after_softmax" : "before_qk";
}

extern "C" int fa_bw_head_dim() { return HEAD_DIM; }
