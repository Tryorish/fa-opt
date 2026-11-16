// Run only on gfx936. Verify the production helper's exact lane/register bits.
#define FA_BW_PROBE_LAYOUT_VERSION 2
#include "matrix_v.h"
#include "async_v.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

static_assert(FA_BW_MATRIX_V == 1, "Compile this probe with -DFA_BW_MATRIX_V=1");
using bits8 = uint16_t __attribute__((ext_vector_type(8)));

#ifndef FA_BW_TEST_KEYS
#define FA_BW_TEST_KEYS 64
#endif
constexpr int Keys = FA_BW_TEST_KEYS;
// Probe V at its highest production LDS offset, including the 64 KiB case.
constexpr int Prefix = 3 * Keys * HEAD_DIM;
static_assert((Prefix + Keys * HEAD_DIM) * 2 <= 65536, "Probe must fit 64 KiB LDS");

__global__ void probe_v_matrix(const uint16_t* input, uint16_t* output) {
    __shared__ __align__(16) uint16_t sm[Prefix + Keys * HEAD_DIM];
    auto* tile = sm + Prefix;
    // Observe the prefix too, retaining the full allocation and checking that
    // direct writes into the high buffer do not damage earlier K/V buffers.
    volatile uint16_t* guard = sm;
    if constexpr (Prefix) {
        for (int i = threadIdx.x; i < Prefix; i += blockDim.x) guard[i] = 0x7bad;
        __syncthreads();
    }
    if constexpr (FA_BW_ASYNC_V) {
        async_v_to_lds<Keys>(reinterpret_cast<const _Float16*>(input), reinterpret_cast<_Float16*>(tile));
        wait_async_v();
    } else {
        for (int i = threadIdx.x; i < Keys * HEAD_DIM; i += blockDim.x)
            tile[lds_v_matrix(i / HEAD_DIM, i % HEAD_DIM)] = uint16_t(0x2000 + i);
    }
    __syncthreads();
    const half8 value = load_v_matrix(reinterpret_cast<_Float16*>(tile), blockIdx.x, blockIdx.y);
    const int base = ((blockIdx.y * (Keys / 16) + blockIdx.x) * 256 + threadIdx.x) * 8;
    // Clang can bit_cast value[r] from the vector's base, ignoring r.
    // Cast/store the whole vector so every returned bit reaches the host.
    bool intact = true;
    if constexpr (Prefix)
        for (int i = threadIdx.x; i < Prefix; i += blockDim.x) intact &= guard[i] == 0x7bad;
    *reinterpret_cast<bits8*>(output + base) = intact ? __builtin_bit_cast(bits8, value) : bits8{};
}

#define CHECK_HIP(call) do { \
    const hipError_t status = (call); \
    if (status != hipSuccess) { \
        std::cerr << #call << ": " << hipGetErrorString(status) << '\n'; \
        return 1; \
    } \
} while (0)

int main() {
    hipDeviceProp_t props{};
    CHECK_HIP(hipGetDeviceProperties(&props, 0));
    if (std::strncmp(props.gcnArchName, "gfx936", 6) != 0 ||
        (props.gcnArchName[6] != '\0' && props.gcnArchName[6] != ':')) {
        std::cerr << "requires gfx936, got " << props.gcnArchName << '\n';
        return 1;
    }
    constexpr int count = (Keys / 16) * (HEAD_DIM / 32) * 256 * 8;
    std::cout << (FA_BW_ASYNC_V ? "Async V" : "Matrix V") << " probe: D=" << HEAD_DIM
              << " C=" << Keys << " panels=" << HEAD_DIM / 32
              << " LDS_bytes=" << (Prefix + Keys * HEAD_DIM) * sizeof(uint16_t)
              << " expected_tags=" << count << std::endl;
    uint16_t* input = nullptr;
    // Offset the source by 16B, checking resource-base addressing as well.
    std::vector<uint16_t> tags(Keys * HEAD_DIM + 8);
    for (int i = 0; i < Keys * HEAD_DIM; ++i) tags[i + 8] = uint16_t(0x2000 + i);
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&input), tags.size() * sizeof(uint16_t)));
    CHECK_HIP(hipMemcpy(input, tags.data(), tags.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    uint16_t* output = nullptr;
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&output), count * sizeof(uint16_t)));
    hipLaunchKernelGGL(probe_v_matrix, dim3(Keys / 16, HEAD_DIM / 32), dim3(256), 0, 0, input + 8, output);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    std::vector<uint16_t> actual(count);
    CHECK_HIP(hipMemcpy(actual.data(), output, count * sizeof(uint16_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(output));
    CHECK_HIP(hipFree(input));
    int failures = 0;
    for (int panel = 0; panel < HEAD_DIM / 32; ++panel)
        for (int k = 0; k < Keys / 16; ++k)
            for (int tid = 0; tid < 256; ++tid)
                for (int r = 0; r < 8; ++r) {
                    const int lane = tid % 64;
                    const int key = k * 16 + (lane / 16) * 4 + r % 4;
                    const int col = panel * 32 + (r / 4) * 16 + lane % 16;
                    const uint16_t expected = uint16_t(0x2000 + key * HEAD_DIM + col);
                    const int i = ((panel * (Keys / 16) + k) * 256 + tid) * 8 + r;
                    if (actual[i] != expected && failures++ < 8)
                        std::cerr << "panel=" << panel << " key_tile=" << k << " tid=" << tid
                                  << " reg=" << r << " expected_bits=" << expected
                                  << " actual_bits=" << actual[i] << '\n';
                }
    std::cout << (FA_BW_ASYNC_V ? "Async V" : "Matrix V") << " tags: "
              << count - failures << '/' << count << " match\n";
    return failures ? 1 : 0;
}
