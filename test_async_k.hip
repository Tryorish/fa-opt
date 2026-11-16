// Exact copy/layout check with multiple source tiles and guards around LDS.
#define FA_BW_PROBE_LAYOUT_VERSION 2
#include "async_k.h"
#include <cstring>
#include <iostream>
#include <vector>

#ifndef FA_BW_TEST_KEYS
#define FA_BW_TEST_KEYS 64
#endif
constexpr int Keys = FA_BW_TEST_KEYS, Elements = Keys * HEAD_DIM;
constexpr int Guard = 8, OutputStride = Elements + 2 * Guard;
static_assert(FA_BW_ASYNC_K == 1, "Enable async K for this probe");

__global__ void probe_async_k(const uint16_t* input, uint16_t* output) {
    __shared__ __align__(16) uint16_t sm[OutputStride];
    for (int i = threadIdx.x; i < OutputStride; i += 256) sm[i] = 0x7bad;
    __syncthreads();
    async_k_to_lds<Keys>(reinterpret_cast<const _Float16*>(input + blockIdx.x * Elements),
                        reinterpret_cast<_Float16*>(sm + Guard));
    wait_async_v();  // The same VMEM/LDS wait completes both direct-copy types.
    __syncthreads();
    output += blockIdx.x * OutputStride;
    for (int i = threadIdx.x; i < OutputStride; i += 256) {
        if (i < Guard || i >= Guard + Elements)
            output[i] = sm[i];
        else {
            const int logical = i - Guard;
            output[i] = sm[Guard + lds<HEAD_DIM>(logical / HEAD_DIM, logical % HEAD_DIM)];
        }
    }
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
        (props.gcnArchName[6] != '\0' && props.gcnArchName[6] != ':')) return 1;
    std::cout << "Async K probe: D=" << HEAD_DIM << " C=" << Keys
              << " elements_per_tile=" << Elements
              << " LDS_bytes=" << OutputStride * sizeof(uint16_t)
              << " expected_tags=" << 2 * Elements << std::endl;
    std::vector<uint16_t> tags(2 * Elements + Guard), actual(2 * OutputStride);
    for (int i = 0; i < 2 * Elements; ++i) tags[Guard + i] = uint16_t(0x1000 + i);
    uint16_t *input = nullptr, *output = nullptr;
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&input), tags.size() * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&output), actual.size() * sizeof(uint16_t)));
    CHECK_HIP(hipMemcpy(input, tags.data(), tags.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe_async_k, dim3(2), dim3(256), 0, 0, input + Guard, output);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(actual.data(), output, actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(input));
    CHECK_HIP(hipFree(output));
    int failures = 0;
    for (int tile = 0; tile < 2; ++tile) {
        for (int i = 0; i < OutputStride; ++i) {
            const bool guard = i < Guard || i >= Guard + Elements;
            const uint16_t expected = guard ? 0x7bad : uint16_t(0x1000 + tile * Elements + i - Guard);
            const uint16_t value = actual[tile * OutputStride + i];
            if (value != expected && failures++ < 8)
                std::cerr << "tile=" << tile << " output_index=" << i
                          << " expected=" << expected << " actual=" << value << '\n';
        }
    }
    std::cout << "Async K C" << Keys << ": " << 2 * Elements << " tags and LDS guards "
              << (failures ? "FAILED" : "match") << '\n';
    return failures ? 1 : 0;
}
