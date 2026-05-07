#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kTargetBlocks = 64;

__global__ void hold_one_sm_kernel(unsigned long long wait_cycles,
                                   unsigned int* block_smids,
                                   unsigned long long* sink) {
    extern __shared__ unsigned int scratch[];

    unsigned int smid = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

    if (threadIdx.x == 0) {
        block_smids[blockIdx.x] = smid;
    }

    unsigned long long state =
        (static_cast<unsigned long long>(smid) << 32) ^
        (0x9e3779b97f4a7c15ULL + static_cast<unsigned long long>(blockIdx.x) * 0x100000001b3ULL +
         static_cast<unsigned long long>(threadIdx.x));

    const unsigned long long start = clock64();
    while ((clock64() - start) < wait_cycles) {
        state = state * 2862933555777941757ULL + 3037000493ULL;
        if (threadIdx.x < 32) {
            scratch[threadIdx.x] = static_cast<unsigned int>(state);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(sink, state);
    }
}

}  // namespace

int main() {
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int clock_khz = 0;
    int shared_mem_per_sm = 0;
    int max_optin_shared_per_block = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&shared_mem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&max_optin_shared_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    const int requested_dynamic_smem = shared_mem_per_sm / 2 + 1024;
    const int dynamic_smem_bytes = std::min(requested_dynamic_smem, max_optin_shared_per_block);

    CHECK_CUDA(cudaFuncSetAttribute(
        hold_one_sm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        hold_one_sm_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    int active_blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, hold_one_sm_kernel, kThreadsPerBlock, dynamic_smem_bytes));

    cudaFuncAttributes attr{};
    CHECK_CUDA(cudaFuncGetAttributes(&attr, hold_one_sm_kernel));

    std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("SM count: %d\n", prop.multiProcessorCount);
    std::printf("Clock rate: %d kHz\n", clock_khz);
    std::printf("Shared mem per SM: %d bytes\n", shared_mem_per_sm);
    std::printf("Max opt-in shared mem per block: %d bytes\n", max_optin_shared_per_block);
    std::printf("Kernel threads per block: %d\n", kThreadsPerBlock);
    std::printf("Kernel static shared mem: %zu bytes\n", attr.sharedSizeBytes);
    std::printf("Kernel num regs per thread: %d\n", attr.numRegs);
    std::printf("Requested dynamic shared mem: %d bytes\n", requested_dynamic_smem);
    std::printf("Configured dynamic shared mem: %d bytes\n", dynamic_smem_bytes);
    std::printf("Occupancy result: active_blocks_per_sm = %d\n", active_blocks_per_sm);
    std::printf("If launch %d blocks, theoretical resident blocks <= min(%d, %d * %d) = %d\n",
                kTargetBlocks,
                kTargetBlocks,
                active_blocks_per_sm,
                prop.multiProcessorCount,
                std::min(kTargetBlocks, active_blocks_per_sm * prop.multiProcessorCount));

    if (active_blocks_per_sm == 1) {
        std::printf("Conclusion: this kernel is 1 block/SM on the current GPU, so %d blocks targets %d SMs.\n",
                    kTargetBlocks, kTargetBlocks);
    } else {
        std::printf("Conclusion: this kernel is NOT 1 block/SM on the current GPU yet.\n");
    }

    return 0;
}
