#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA Runtime error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

#define CHECK_CU(call) do { \
    CUresult err__ = (call); \
    if (err__ != CUDA_SUCCESS) { \
        const char* err_str__ = nullptr; \
        cuGetErrorString(err__, &err_str__); \
        std::fprintf(stderr, "CUDA Driver error at %s:%d: %s\n", __FILE__, __LINE__, err_str__ ? err_str__ : "unknown"); \
        std::exit(1); \
    } \
} while (0)

namespace {

constexpr int kDefaultGcSms = 56;
constexpr int kDefaultLaunchSms = 56;
constexpr int kThreadsPerBlock = 256;
constexpr int kDefaultWorkIters = 1 << 20;
constexpr int kDefaultWarmupRounds = 5;
constexpr int kDefaultMeasureRounds = 5;

// IO kernel defaults
constexpr unsigned long long kDefaultIoElements = 16ULL * 1024ULL * 1024ULL;  // 16M floats = 64MB
constexpr int kDefaultIoIters = 4;
constexpr int kDefaultIoBlocks = 8;
constexpr int kDefaultIoKernelCount = 4;

struct DeviceCaps {
    int device = 0;
    int sm_count = 0;
    int major = 0;
    int minor = 0;
    int clock_khz = 0;
    int shared_mem_per_sm = 0;
    int max_optin_shared_per_block = 0;
    char name[256] = {};
};

struct LaunchConfig {
    int blocks = kDefaultLaunchSms;
    int threads = kThreadsPerBlock;
    int dynamic_smem_bytes = 0;
    int work_iters = kDefaultWorkIters;
};

struct IoLaunchConfig {
    int blocks = kDefaultIoBlocks;
    int threads = kThreadsPerBlock;
    unsigned long long num_elements = kDefaultIoElements;
    unsigned long long io_iters = kDefaultIoIters;
};

struct KernelReport {
    std::string label;
    float event_ms = 0.0f;
    double wall_ms = 0.0;
    std::vector<unsigned int> block_smids;
    std::set<unsigned int> sm_set;
    unsigned long long sink_value = 0;
};

struct RoundReport {
    KernelReport a;
    KernelReport b;
    KernelReport c;
    bool has_c = false;
    double case_wall_ms = 0.0;
};

struct BigIoRoundReport {
    KernelReport big;
    std::vector<KernelReport> io_reports;
    double case_wall_ms = 0.0;
};

struct LaunchGate {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
};

struct GcResources {
    CUgreenCtx green_ctx = nullptr;
    CUcontext cuda_ctx = nullptr;
    unsigned int sm_count = 0;
};

__device__ __forceinline__ unsigned int get_smid() {
    unsigned int smid = 0;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__global__ void hold_one_sm_kernel(unsigned long long work_iters,
                                   unsigned int* block_smids,
                                   unsigned long long* sink) {
    extern __shared__ unsigned int scratch[];

    const int lane = threadIdx.x & 31;
    const unsigned int smid = get_smid();

    if (threadIdx.x == 0) {
        block_smids[blockIdx.x] = smid;
    }

    unsigned long long state =
        (static_cast<unsigned long long>(smid) << 32) +
        (0x9e3779b97f4a7c15ULL + static_cast<unsigned long long>(blockIdx.x) * 0x100000001b3ULL +
         static_cast<unsigned long long>(threadIdx.x));

    #pragma unroll 1
    for (unsigned long long iter = 0; iter < work_iters; ++iter) {
        state = state * 2862933555777941757ULL + 3037000493ULL;
        if (threadIdx.x < 32) {
            scratch[lane] = static_cast<unsigned int>(state);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(sink, state);
    }
}

// IO-intensive kernel: pure global memory streaming, minimal compute.
// Bottleneck is HBM bandwidth, not SM compute units.
__global__ void io_intensive_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    unsigned long long num_elements,
                                    unsigned long long io_iters,
                                    unsigned int* block_smids,
                                    unsigned long long* sink) {
    const unsigned int smid = get_smid();
    if (threadIdx.x == 0) {
        block_smids[blockIdx.x] = smid;
    }

    const unsigned long long tid = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    float acc = 0.0f;

    #pragma unroll 1
    for (unsigned long long iter = 0; iter < io_iters; ++iter) {
        for (unsigned long long i = tid; i < num_elements; i += stride) {
            acc += src[i];
            dst[i] = acc;
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(sink, static_cast<unsigned long long>(__float_as_uint(acc)));
    }
}

DeviceCaps query_device_caps(int device) {
    DeviceCaps caps;
    caps.device = device;

    CHECK_CU(cuInit(0));

    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, device));
    CHECK_CU(cuDeviceGetName(caps.name, sizeof(caps.name), cu_device));
    CHECK_CU(cuDeviceGetAttribute(&caps.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_device));
    CHECK_CU(cuDeviceGetAttribute(&caps.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device));
    CHECK_CU(cuDeviceGetAttribute(&caps.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device));

    CHECK_CUDA(cudaSetDevice(device));
    CHECK_CUDA(cudaDeviceGetAttribute(&caps.clock_khz, cudaDevAttrClockRate, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&caps.shared_mem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&caps.max_optin_shared_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

    return caps;
}

LaunchConfig build_launch_config(const DeviceCaps& caps, int work_iters, int launch_sms) {
    LaunchConfig cfg;
    cfg.blocks = launch_sms;
    cfg.work_iters = work_iters;

    const int requested_smem = caps.shared_mem_per_sm / 2 + 1024;
    cfg.dynamic_smem_bytes = std::min(requested_smem, caps.max_optin_shared_per_block);

    if (cfg.dynamic_smem_bytes <= caps.shared_mem_per_sm / 2) {
        std::fprintf(stderr,
                     "Unable to force 1 block/SM via dynamic shared memory: per_sm=%d, optin_per_block=%d\n",
                     caps.shared_mem_per_sm, caps.max_optin_shared_per_block);
        std::exit(1);
    }

    CHECK_CUDA(cudaFuncSetAttribute(
        hold_one_sm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.dynamic_smem_bytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        hold_one_sm_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    int active_blocks_per_sm = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, hold_one_sm_kernel, cfg.threads, cfg.dynamic_smem_bytes));

    if (active_blocks_per_sm != 1) {
        std::fprintf(stderr,
                     "Expected 1 block/SM but occupancy API returned %d (smem=%d bytes)\n",
                     active_blocks_per_sm, cfg.dynamic_smem_bytes);
        std::exit(1);
    }

    return cfg;
}

std::set<unsigned int> make_sm_set(const std::vector<unsigned int>& smids) {
    std::set<unsigned int> out;
    for (unsigned int smid : smids) {
        out.insert(smid);
    }
    return out;
}

void run_primary_context_task(const DeviceCaps& caps,
                              const LaunchConfig& cfg,
                              const char* label,
                              LaunchGate* gate,
                              KernelReport* report) {
    report->label = label;

    CHECK_CUDA(cudaSetDevice(caps.device));

    cudaStream_t stream;
    cudaEvent_t start_evt, stop_evt;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&start_evt));
    CHECK_CUDA(cudaEventCreate(&stop_evt));

    unsigned int* d_smids = nullptr;
    unsigned long long* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_smids, cfg.blocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemsetAsync(d_smids, 0xff, cfg.blocks * sizeof(unsigned int), stream));
    CHECK_CUDA(cudaMemsetAsync(d_sink, 0, sizeof(unsigned long long), stream));

    gate->ready.fetch_add(1, std::memory_order_release);
    while (!gate->go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto host_begin = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(start_evt, stream));
    hold_one_sm_kernel<<<cfg.blocks, cfg.threads, cfg.dynamic_smem_bytes, stream>>>(
        static_cast<unsigned long long>(cfg.work_iters), d_smids, d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_evt, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto host_end = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaEventElapsedTime(&report->event_ms, start_evt, stop_evt));
    report->wall_ms =
        std::chrono::duration<double, std::milli>(host_end - host_begin).count();

    report->block_smids.resize(cfg.blocks);
    CHECK_CUDA(cudaMemcpy(report->block_smids.data(), d_smids, cfg.blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&report->sink_value, d_sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    report->sm_set = make_sm_set(report->block_smids);

    CHECK_CUDA(cudaFree(d_smids));
    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaEventDestroy(start_evt));
    CHECK_CUDA(cudaEventDestroy(stop_evt));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

void run_primary_context_io_task(const DeviceCaps& caps,
                                 const IoLaunchConfig& cfg,
                                 const char* label,
                                 LaunchGate* gate,
                                 KernelReport* report) {
    report->label = label;

    CHECK_CUDA(cudaSetDevice(caps.device));

    cudaStream_t stream;
    cudaEvent_t start_evt, stop_evt;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaEventCreate(&start_evt));
    CHECK_CUDA(cudaEventCreate(&stop_evt));

    unsigned int* d_smids = nullptr;
    unsigned long long* d_sink = nullptr;
    float* d_src = nullptr;
    float* d_dst = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(cfg.num_elements) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_smids, cfg.blocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemsetAsync(d_smids, 0xff, cfg.blocks * sizeof(unsigned int), stream));
    CHECK_CUDA(cudaMemsetAsync(d_sink, 0, sizeof(unsigned long long), stream));
    CHECK_CUDA(cudaMemsetAsync(d_src, 0, bytes, stream));
    CHECK_CUDA(cudaMemsetAsync(d_dst, 0, bytes, stream));

    gate->ready.fetch_add(1, std::memory_order_release);
    while (!gate->go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto host_begin = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(start_evt, stream));
    io_intensive_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
        d_src, d_dst, cfg.num_elements, cfg.io_iters, d_smids, d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_evt, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto host_end = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaEventElapsedTime(&report->event_ms, start_evt, stop_evt));
    report->wall_ms =
        std::chrono::duration<double, std::milli>(host_end - host_begin).count();

    report->block_smids.resize(cfg.blocks);
    CHECK_CUDA(cudaMemcpy(report->block_smids.data(), d_smids, cfg.blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&report->sink_value, d_sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    report->sm_set = make_sm_set(report->block_smids);

    CHECK_CUDA(cudaFree(d_smids));
    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start_evt));
    CHECK_CUDA(cudaEventDestroy(stop_evt));
    CHECK_CUDA(cudaStreamDestroy(stream));
}


std::vector<GcResources> create_two_green_contexts(int device_id, unsigned int sms_a, unsigned int sms_b) {
    CUdevice cu_device;
    CHECK_CU(cuDeviceGet(&cu_device, device_id));

    CUdevResource full_sm_resource{};
    CHECK_CU(cuDeviceGetDevResource(cu_device, &full_sm_resource, CU_DEV_RESOURCE_TYPE_SM));

    const unsigned int grain = full_sm_resource.sm.minSmPartitionSize;  // 8 on Hopper
    const unsigned int alignment = full_sm_resource.sm.smCoscheduledAlignment;
    std::printf("SM resource: total=%u, minPartition=%u, coscheduledAlignment=%u\n",
                full_sm_resource.sm.smCount, grain, alignment);

    if (grain == 0) {
        std::fprintf(stderr, "minSmPartitionSize is 0, cannot partition\n");
        std::exit(1);
    }

    // Align requested sizes up to alignment boundary
    const unsigned int actual_a = ((sms_a + alignment - 1) / alignment) * alignment;
    const unsigned int actual_b = ((sms_b + alignment - 1) / alignment) * alignment;

    if (actual_a + actual_b > full_sm_resource.sm.smCount) {
        std::fprintf(stderr, "ERROR: aligned gc-sms-a(%u) + gc-sms-b(%u) = %u > device SMs(%u)\n",
                     actual_a, actual_b, actual_a + actual_b, full_sm_resource.sm.smCount);
        std::exit(1);
    }

    // Use grain (minSmPartitionSize) as chunk size for maximum flexibility
    const unsigned int chunk_size = grain;
    const unsigned int chunks_a = actual_a / chunk_size;
    const unsigned int chunks_b = actual_b / chunk_size;
    const unsigned int total_chunks = chunks_a + chunks_b;

    // Dry-run: query how many groups driver will actually create
    unsigned int dry_run_count = total_chunks;
    CHECK_CU(cuDevSmResourceSplitByCount(nullptr, &dry_run_count, &full_sm_resource, nullptr, 0, chunk_size));

    std::printf("GC partition: chunk_size=%u, requested=%u chunks, driver can create=%u chunks\n",
                chunk_size, total_chunks, dry_run_count);

    if (dry_run_count < total_chunks) {
        std::fprintf(stderr, "Driver can only create %u groups of %u+ SMs, but we need %u\n",
                     dry_run_count, chunk_size, total_chunks);
        std::exit(1);
    }

    // Actual split
    std::vector<CUdevResource> chunks(dry_run_count);
    CUdevResource remaining{};
    unsigned int group_count = dry_run_count;
    CHECK_CU(cuDevSmResourceSplitByCount(chunks.data(), &group_count, &full_sm_resource, &remaining, 0, chunk_size));

    std::printf("Split result: %u groups, each %u SMs, remaining %u SMs\n",
                group_count, chunks[0].sm.smCount, remaining.sm.smCount);

    // Merge chunks [0, chunks_a) for GC_A and [chunks_a, chunks_a+chunks_b) for GC_B
    std::vector<GcResources> out(2);

    {
        CUdevResourceDesc desc_a = nullptr;
        CHECK_CU(cuDevResourceGenerateDesc(&desc_a, &chunks[0], chunks_a));
        CHECK_CU(cuGreenCtxCreate(&out[0].green_ctx, desc_a, cu_device, CU_GREEN_CTX_DEFAULT_STREAM));
        CHECK_CU(cuCtxFromGreenCtx(&out[0].cuda_ctx, out[0].green_ctx));
        // Sum up actual SM count from chunks
        unsigned int sum_a = 0;
        for (unsigned int i = 0; i < chunks_a; ++i) sum_a += chunks[i].sm.smCount;
        out[0].sm_count = sum_a;
    }

    {
        CUdevResourceDesc desc_b = nullptr;
        CHECK_CU(cuDevResourceGenerateDesc(&desc_b, &chunks[chunks_a], chunks_b));
        CHECK_CU(cuGreenCtxCreate(&out[1].green_ctx, desc_b, cu_device, CU_GREEN_CTX_DEFAULT_STREAM));
        CHECK_CU(cuCtxFromGreenCtx(&out[1].cuda_ctx, out[1].green_ctx));
        unsigned int sum_b = 0;
        for (unsigned int i = chunks_a; i < chunks_a + chunks_b; ++i) sum_b += chunks[i].sm.smCount;
        out[1].sm_count = sum_b;
    }

    return out;
}

void run_green_context_task(const DeviceCaps& caps,
                            const LaunchConfig& cfg,
                            const char* label,
                            const GcResources* gc,
                            LaunchGate* gate,
                            KernelReport* report) {
    report->label = label;

    CHECK_CU(cuCtxSetCurrent(gc->cuda_ctx));
    CHECK_CUDA(cudaSetDevice(caps.device));

    CHECK_CUDA(cudaFuncSetAttribute(
        hold_one_sm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cfg.dynamic_smem_bytes));
    CHECK_CUDA(cudaFuncSetAttribute(
        hold_one_sm_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    CUstream cu_stream = nullptr;
    CHECK_CU(cuGreenCtxStreamCreate(&cu_stream, gc->green_ctx, CU_STREAM_NON_BLOCKING, 0));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cu_stream);

    cudaEvent_t start_evt, stop_evt;
    CHECK_CUDA(cudaEventCreate(&start_evt));
    CHECK_CUDA(cudaEventCreate(&stop_evt));

    unsigned int* d_smids = nullptr;
    unsigned long long* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_smids, cfg.blocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemsetAsync(d_smids, 0xff, cfg.blocks * sizeof(unsigned int), stream));
    CHECK_CUDA(cudaMemsetAsync(d_sink, 0, sizeof(unsigned long long), stream));

    gate->ready.fetch_add(1, std::memory_order_release);
    while (!gate->go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto host_begin = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(start_evt, stream));
    hold_one_sm_kernel<<<cfg.blocks, cfg.threads, cfg.dynamic_smem_bytes, stream>>>(
        static_cast<unsigned long long>(cfg.work_iters), d_smids, d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_evt, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto host_end = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaEventElapsedTime(&report->event_ms, start_evt, stop_evt));
    report->wall_ms =
        std::chrono::duration<double, std::milli>(host_end - host_begin).count();

    report->block_smids.resize(cfg.blocks);
    CHECK_CUDA(cudaMemcpy(report->block_smids.data(), d_smids, cfg.blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&report->sink_value, d_sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    report->sm_set = make_sm_set(report->block_smids);

    CHECK_CUDA(cudaFree(d_smids));
    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaEventDestroy(start_evt));
    CHECK_CUDA(cudaEventDestroy(stop_evt));
    CHECK_CU(cuStreamDestroy(cu_stream));
}

void run_green_context_io_task(const DeviceCaps& caps,
                               const IoLaunchConfig& cfg,
                               const char* label,
                               const GcResources* gc,
                               LaunchGate* gate,
                               KernelReport* report) {
    report->label = label;

    CHECK_CU(cuCtxSetCurrent(gc->cuda_ctx));
    CHECK_CUDA(cudaSetDevice(caps.device));

    CUstream cu_stream = nullptr;
    CHECK_CU(cuGreenCtxStreamCreate(&cu_stream, gc->green_ctx, CU_STREAM_NON_BLOCKING, 0));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(cu_stream);

    cudaEvent_t start_evt, stop_evt;
    CHECK_CUDA(cudaEventCreate(&start_evt));
    CHECK_CUDA(cudaEventCreate(&stop_evt));

    unsigned int* d_smids = nullptr;
    unsigned long long* d_sink = nullptr;
    float* d_src = nullptr;
    float* d_dst = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(cfg.num_elements) * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_smids, cfg.blocks * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_src, bytes));
    CHECK_CUDA(cudaMalloc(&d_dst, bytes));
    CHECK_CUDA(cudaMemsetAsync(d_smids, 0xff, cfg.blocks * sizeof(unsigned int), stream));
    CHECK_CUDA(cudaMemsetAsync(d_sink, 0, sizeof(unsigned long long), stream));
    CHECK_CUDA(cudaMemsetAsync(d_src, 0, bytes, stream));
    CHECK_CUDA(cudaMemsetAsync(d_dst, 0, bytes, stream));

    gate->ready.fetch_add(1, std::memory_order_release);
    while (!gate->go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto host_begin = std::chrono::steady_clock::now();
    CHECK_CUDA(cudaEventRecord(start_evt, stream));
    io_intensive_kernel<<<cfg.blocks, cfg.threads, 0, stream>>>(
        d_src, d_dst, cfg.num_elements, cfg.io_iters, d_smids, d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_evt, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    const auto host_end = std::chrono::steady_clock::now();

    CHECK_CUDA(cudaEventElapsedTime(&report->event_ms, start_evt, stop_evt));
    report->wall_ms =
        std::chrono::duration<double, std::milli>(host_end - host_begin).count();

    report->block_smids.resize(cfg.blocks);
    CHECK_CUDA(cudaMemcpy(report->block_smids.data(), d_smids, cfg.blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&report->sink_value, d_sink, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    report->sm_set = make_sm_set(report->block_smids);

    CHECK_CUDA(cudaFree(d_smids));
    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaEventDestroy(start_evt));
    CHECK_CUDA(cudaEventDestroy(stop_evt));
    CHECK_CU(cuStreamDestroy(cu_stream));
}

RoundReport run_primary_context_pair_once(const DeviceCaps& caps,
                                          const LaunchConfig& cfg_a,
                                          const LaunchConfig& cfg_b,
                                          const char* label_a,
                                          const char* label_b) {
    LaunchGate gate;
    RoundReport round;
    std::thread t0(run_primary_context_task, std::cref(caps), std::cref(cfg_a),
                   label_a, &gate, &round.a);
    std::thread t1(run_primary_context_task, std::cref(caps), std::cref(cfg_b),
                   label_b, &gate, &round.b);

    while (gate.ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }

    const auto case_begin = std::chrono::steady_clock::now();
    gate.go.store(true, std::memory_order_release);
    t0.join();
    t1.join();
    const auto case_end = std::chrono::steady_clock::now();

    round.case_wall_ms =
        std::chrono::duration<double, std::milli>(case_end - case_begin).count();
    return round;
}

RoundReport run_primary_context_triple_once(const DeviceCaps& caps,
                                            const LaunchConfig& cfg_a,
                                            const LaunchConfig& cfg_b,
                                            const LaunchConfig& cfg_c,
                                            const char* label_a,
                                            const char* label_b,
                                            const char* label_c) {
    LaunchGate gate;
    RoundReport round;
    round.has_c = true;
    std::thread t0(run_primary_context_task, std::cref(caps), std::cref(cfg_a),
                   label_a, &gate, &round.a);
    std::thread t1(run_primary_context_task, std::cref(caps), std::cref(cfg_b),
                   label_b, &gate, &round.b);
    std::thread t2(run_primary_context_task, std::cref(caps), std::cref(cfg_c),
                   label_c, &gate, &round.c);

    while (gate.ready.load(std::memory_order_acquire) != 3) {
        std::this_thread::yield();
    }

    const auto case_begin = std::chrono::steady_clock::now();
    gate.go.store(true, std::memory_order_release);
    t0.join();
    t1.join();
    t2.join();
    const auto case_end = std::chrono::steady_clock::now();

    round.case_wall_ms =
        std::chrono::duration<double, std::milli>(case_end - case_begin).count();
    return round;
}

RoundReport run_green_context_pair_once(const DeviceCaps& caps,
                                        const LaunchConfig& cfg_a,
                                        const LaunchConfig& cfg_b,
                                        const std::vector<GcResources>& gc_resources,
                                        const char* label_a,
                                        const char* label_b) {
    LaunchGate gate;
    RoundReport round;
    std::thread t0(run_green_context_task, std::cref(caps), std::cref(cfg_a),
                   label_a, &gc_resources[0], &gate, &round.a);
    std::thread t1(run_green_context_task, std::cref(caps), std::cref(cfg_b),
                   label_b, &gc_resources[1], &gate, &round.b);

    while (gate.ready.load(std::memory_order_acquire) != 2) {
        std::this_thread::yield();
    }

    const auto case_begin = std::chrono::steady_clock::now();
    gate.go.store(true, std::memory_order_release);
    t0.join();
    t1.join();
    const auto case_end = std::chrono::steady_clock::now();

    round.case_wall_ms =
        std::chrono::duration<double, std::milli>(case_end - case_begin).count();
    return round;
}

RoundReport run_green_context_triple_once(const DeviceCaps& caps,
                                          const LaunchConfig& cfg_a,
                                          const LaunchConfig& cfg_b,
                                          const LaunchConfig& cfg_c,
                                          const std::vector<GcResources>& gc_resources,
                                          const char* label_a,
                                          const char* label_b,
                                          const char* label_c) {
    LaunchGate gate;
    RoundReport round;
    round.has_c = true;
    std::thread t0(run_green_context_task, std::cref(caps), std::cref(cfg_a),
                   label_a, &gc_resources[0], &gate, &round.a);
    std::thread t1(run_green_context_task, std::cref(caps), std::cref(cfg_b),
                   label_b, &gc_resources[1], &gate, &round.b);
    std::thread t2(run_green_context_task, std::cref(caps), std::cref(cfg_c),
                   label_c, &gc_resources[1], &gate, &round.c);

    while (gate.ready.load(std::memory_order_acquire) != 3) {
        std::this_thread::yield();
    }

    const auto case_begin = std::chrono::steady_clock::now();
    gate.go.store(true, std::memory_order_release);
    t0.join();
    t1.join();
    t2.join();
    const auto case_end = std::chrono::steady_clock::now();

    round.case_wall_ms =
        std::chrono::duration<double, std::milli>(case_end - case_begin).count();
    return round;
}

BigIoRoundReport run_primary_context_big_io_once(const DeviceCaps& caps,
                                                 const LaunchConfig& big_cfg,
                                                 const IoLaunchConfig& io_cfg,
                                                 int io_kernel_count) {
    LaunchGate gate;
    BigIoRoundReport round;
    round.io_reports.resize(io_kernel_count);

    std::thread big_thread(run_primary_context_task, std::cref(caps), std::cref(big_cfg),
                           "baseline-big", &gate, &round.big);

    std::vector<std::string> io_labels(io_kernel_count);
    std::vector<std::thread> io_threads;
    io_threads.reserve(io_kernel_count);

    for (int i = 0; i < io_kernel_count; ++i) {
        io_labels[i] = "baseline-io-" + std::to_string(i);
        io_threads.emplace_back(run_primary_context_io_task, std::cref(caps), std::cref(io_cfg),
                                io_labels[i].c_str(), &gate, &round.io_reports[i]);
    }

    while (gate.ready.load(std::memory_order_acquire) != io_kernel_count + 1) {
        std::this_thread::yield();
    }

    const auto case_begin = std::chrono::steady_clock::now();
    gate.go.store(true, std::memory_order_release);
    big_thread.join();
    for (std::thread& t : io_threads) {
        t.join();
    }
    const auto case_end = std::chrono::steady_clock::now();

    round.case_wall_ms =
        std::chrono::duration<double, std::milli>(case_end - case_begin).count();
    return round;
}

BigIoRoundReport run_green_context_big_io_once(const DeviceCaps& caps,
                                               const LaunchConfig& big_cfg,
                                               const IoLaunchConfig& io_cfg,
                                               int io_kernel_count,
                                               const std::vector<GcResources>& gc_resources) {
    LaunchGate gate;
    BigIoRoundReport round;
    round.io_reports.resize(io_kernel_count);

    std::thread big_thread(run_green_context_task, std::cref(caps), std::cref(big_cfg),
                           "gc-big", &gc_resources[0], &gate, &round.big);

    std::vector<std::string> io_labels(io_kernel_count);
    std::vector<std::thread> io_threads;
    io_threads.reserve(io_kernel_count);

    for (int i = 0; i < io_kernel_count; ++i) {
        io_labels[i] = "gc-io-" + std::to_string(i);
        io_threads.emplace_back(run_green_context_io_task, std::cref(caps), std::cref(io_cfg),
                                io_labels[i].c_str(), &gc_resources[1], &gate, &round.io_reports[i]);
    }

    while (gate.ready.load(std::memory_order_acquire) != io_kernel_count + 1) {
        std::this_thread::yield();
    }

    const auto case_begin = std::chrono::steady_clock::now();
    gate.go.store(true, std::memory_order_release);
    big_thread.join();
    for (std::thread& t : io_threads) {
        t.join();
    }
    const auto case_end = std::chrono::steady_clock::now();

    round.case_wall_ms =
        std::chrono::duration<double, std::milli>(case_end - case_begin).count();
    return round;
}

double average_case_wall(const std::vector<RoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const RoundReport& r : rounds) {
        total += r.case_wall_ms;
    }
    return total / static_cast<double>(rounds.size());
}

double average_event_a(const std::vector<RoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const RoundReport& r : rounds) {
        total += r.a.event_ms;
    }
    return total / static_cast<double>(rounds.size());
}

double average_event_b(const std::vector<RoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const RoundReport& r : rounds) {
        total += r.b.event_ms;
    }
    return total / static_cast<double>(rounds.size());
}

double average_event_c(const std::vector<RoundReport>& rounds) {
    if (rounds.empty() || !rounds.front().has_c) {
        return 0.0;
    }
    double total = 0.0;
    for (const RoundReport& r : rounds) {
        total += r.c.event_ms;
    }
    return total / static_cast<double>(rounds.size());
}

double io_avg_event(const BigIoRoundReport& round) {
    if (round.io_reports.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const KernelReport& report : round.io_reports) {
        total += report.event_ms;
    }
    return total / static_cast<double>(round.io_reports.size());
}

double io_max_event(const BigIoRoundReport& round) {
    double max_event = 0.0;
    for (const KernelReport& report : round.io_reports) {
        max_event = std::max(max_event, static_cast<double>(report.event_ms));
    }
    return max_event;
}

double io_sum_event(const BigIoRoundReport& round) {
    double total = 0.0;
    for (const KernelReport& report : round.io_reports) {
        total += report.event_ms;
    }
    return total;
}

double average_big_io_case_wall(const std::vector<BigIoRoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const BigIoRoundReport& round : rounds) {
        total += round.case_wall_ms;
    }
    return total / static_cast<double>(rounds.size());
}

double average_big_event(const std::vector<BigIoRoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const BigIoRoundReport& round : rounds) {
        total += round.big.event_ms;
    }
    return total / static_cast<double>(rounds.size());
}

double average_io_avg_event(const std::vector<BigIoRoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const BigIoRoundReport& round : rounds) {
        total += io_avg_event(round);
    }
    return total / static_cast<double>(rounds.size());
}

double average_io_max_event(const std::vector<BigIoRoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const BigIoRoundReport& round : rounds) {
        total += io_max_event(round);
    }
    return total / static_cast<double>(rounds.size());
}

double average_io_sum_event(const std::vector<BigIoRoundReport>& rounds) {
    if (rounds.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const BigIoRoundReport& round : rounds) {
        total += io_sum_event(round);
    }
    return total / static_cast<double>(rounds.size());
}

void print_measurement_summary(const char* title,
                               const std::vector<RoundReport>& rounds) {
    std::printf("\n=== %s ===\n", title);
    const bool has_c = !rounds.empty() && rounds.front().has_c;
    for (size_t i = 0; i < rounds.size(); ++i) {
        const RoundReport& r = rounds[i];
        if (has_c) {
            std::printf("round %zu: case_wall=%.3f ms, A_event=%.3f ms, B_event=%.3f ms, C_event=%.3f ms, A_wall=%.3f ms, B_wall=%.3f ms, C_wall=%.3f ms\n",
                        i + 1, r.case_wall_ms, r.a.event_ms, r.b.event_ms, r.c.event_ms,
                        r.a.wall_ms, r.b.wall_ms, r.c.wall_ms);
        } else {
            std::printf("round %zu: case_wall=%.3f ms, A_event=%.3f ms, B_event=%.3f ms, A_wall=%.3f ms, B_wall=%.3f ms\n",
                        i + 1, r.case_wall_ms, r.a.event_ms, r.b.event_ms, r.a.wall_ms, r.b.wall_ms);
        }
    }
    if (has_c) {
        std::printf("avg: case_wall=%.3f ms, A_event=%.3f ms, B_event=%.3f ms, C_event=%.3f ms\n",
                    average_case_wall(rounds), average_event_a(rounds), average_event_b(rounds), average_event_c(rounds));
    } else {
        std::printf("avg: case_wall=%.3f ms, A_event=%.3f ms, B_event=%.3f ms\n",
                    average_case_wall(rounds), average_event_a(rounds), average_event_b(rounds));
    }
}

void print_big_io_measurement_summary(const char* title,
                                      const std::vector<BigIoRoundReport>& rounds) {
    std::printf("\n=== %s ===\n", title);
    for (size_t i = 0; i < rounds.size(); ++i) {
        const BigIoRoundReport& round = rounds[i];
        std::printf("round %zu: case_wall=%.3f ms, big_event=%.3f ms, big_wall=%.3f ms, io_avg_event=%.3f ms, io_max_event=%.3f ms, io_sum_event=%.3f ms\n",
                    i + 1,
                    round.case_wall_ms,
                    round.big.event_ms,
                    round.big.wall_ms,
                    io_avg_event(round),
                    io_max_event(round),
                    io_sum_event(round));
    }

    std::printf("avg: case_wall=%.3f ms, big_event=%.3f ms, io_avg_event=%.3f ms, io_max_event=%.3f ms, io_sum_event=%.3f ms\n",
                average_big_io_case_wall(rounds),
                average_big_event(rounds),
                average_io_avg_event(rounds),
                average_io_max_event(rounds),
                average_io_sum_event(rounds));
}

void print_comparison_summary(const std::vector<RoundReport>& baseline_rounds,
                              const std::vector<RoundReport>& gc_rounds) {
    const double baseline_case = average_case_wall(baseline_rounds);
    const double gc_case = average_case_wall(gc_rounds);
    const double baseline_a = average_event_a(baseline_rounds);
    const double baseline_b = average_event_b(baseline_rounds);
    const double baseline_c = average_event_c(baseline_rounds);
    const double gc_a = average_event_a(gc_rounds);
    const double gc_b = average_event_b(gc_rounds);
    const double gc_c = average_event_c(gc_rounds);
    const bool has_c = !baseline_rounds.empty() && baseline_rounds.front().has_c;

    std::printf("\n=== Comparison Summary ===\n");
    std::printf("baseline avg case wall: %.3f ms\n", baseline_case);
    std::printf("green context avg case wall: %.3f ms\n", gc_case);
    if (baseline_case > 0.0) {
        std::printf("case wall ratio (gc / baseline): %.3f\n", gc_case / baseline_case);
    }
    if (has_c) {
        std::printf("baseline avg kernel event: A=%.3f ms, B=%.3f ms, C=%.3f ms\n", baseline_a, baseline_b, baseline_c);
        std::printf("green context avg kernel event: A=%.3f ms, B=%.3f ms, C=%.3f ms\n", gc_a, gc_b, gc_c);
        if (baseline_a > 0.0 && baseline_b > 0.0 && baseline_c > 0.0) {
            std::printf("kernel event ratio (gc / baseline): A=%.3f, B=%.3f, C=%.3f\n",
                        gc_a / baseline_a, gc_b / baseline_b, gc_c / baseline_c);
        }
    } else {
        std::printf("baseline avg kernel event: A=%.3f ms, B=%.3f ms\n", baseline_a, baseline_b);
        std::printf("green context avg kernel event: A=%.3f ms, B=%.3f ms\n", gc_a, gc_b);
        if (baseline_a > 0.0 && baseline_b > 0.0) {
            std::printf("kernel event ratio (gc / baseline): A=%.3f, B=%.3f\n",
                        gc_a / baseline_a, gc_b / baseline_b);
        }
    }
}

void print_big_io_comparison_summary(const std::vector<BigIoRoundReport>& baseline_rounds,
                                     const std::vector<BigIoRoundReport>& gc_rounds) {
    const double baseline_case = average_big_io_case_wall(baseline_rounds);
    const double gc_case = average_big_io_case_wall(gc_rounds);
    const double baseline_big = average_big_event(baseline_rounds);
    const double gc_big = average_big_event(gc_rounds);
    const double baseline_io_avg = average_io_avg_event(baseline_rounds);
    const double gc_io_avg = average_io_avg_event(gc_rounds);
    const double baseline_io_max = average_io_max_event(baseline_rounds);
    const double gc_io_max = average_io_max_event(gc_rounds);
    const double baseline_io_sum = average_io_sum_event(baseline_rounds);
    const double gc_io_sum = average_io_sum_event(gc_rounds);

    std::printf("\n=== Big + IO Comparison Summary ===\n");
    std::printf("baseline avg case wall: %.3f ms\n", baseline_case);
    std::printf("green context avg case wall: %.3f ms\n", gc_case);
    if (baseline_case > 0.0) {
        std::printf("case wall ratio (gc / baseline): %.3f\n", gc_case / baseline_case);
    }

    std::printf("baseline avg big event: %.3f ms\n", baseline_big);
    std::printf("green context avg big event: %.3f ms\n", gc_big);
    if (baseline_big > 0.0) {
        std::printf("big event ratio (gc / baseline): %.3f\n", gc_big / baseline_big);
    }

    std::printf("baseline avg io event: mean=%.3f ms, max=%.3f ms, sum=%.3f ms\n",
                baseline_io_avg, baseline_io_max, baseline_io_sum);
    std::printf("green context avg io event: mean=%.3f ms, max=%.3f ms, sum=%.3f ms\n",
                gc_io_avg, gc_io_max, gc_io_sum);
    if (baseline_io_avg > 0.0 && baseline_io_max > 0.0 && baseline_io_sum > 0.0) {
        std::printf("io event ratio (gc / baseline): mean=%.3f, max=%.3f, sum=%.3f\n",
                    gc_io_avg / baseline_io_avg,
                    gc_io_max / baseline_io_max,
                    gc_io_sum / baseline_io_sum);
    }
}

void warmup_primary_context_pair(const DeviceCaps& caps,
                                 const LaunchConfig& cfg_a,
                                 const LaunchConfig& cfg_b,
                                 int rounds) {
    std::printf("\nWarming up Case 1 (normal streams), rounds=%d...\n", rounds);

    for (int round = 0; round < rounds; ++round) {
        (void)run_primary_context_pair_once(caps, cfg_a, cfg_b, "baseline-warmup-A", "baseline-warmup-B");
        std::printf("  warmup round %d/%d done\n", round + 1, rounds);
    }

    std::printf("Warmup done: normal streams.\n");
}

void warmup_primary_context_triple(const DeviceCaps& caps,
                                   const LaunchConfig& cfg_a,
                                   const LaunchConfig& cfg_b,
                                   const LaunchConfig& cfg_c,
                                   int rounds) {
    std::printf("\nWarming up Case 1 (three normal streams), rounds=%d...\n", rounds);

    for (int round = 0; round < rounds; ++round) {
        (void)run_primary_context_triple_once(
            caps, cfg_a, cfg_b, cfg_c, "baseline-warmup-A", "baseline-warmup-B", "baseline-warmup-C");
        std::printf("  warmup round %d/%d done\n", round + 1, rounds);
    }

    std::printf("Warmup done: three normal streams.\n");
}

void warmup_green_context_pair(const DeviceCaps& caps,
                               const LaunchConfig& cfg_a,
                               const LaunchConfig& cfg_b,
                               const std::vector<GcResources>& gc_resources,
                               int rounds) {
    std::printf("\nWarming up Case 2 (%u/%u Green Context streams), rounds=%d...\n",
                gc_resources[0].sm_count, gc_resources[1].sm_count, rounds);

    for (int round = 0; round < rounds; ++round) {
        (void)run_green_context_pair_once(caps, cfg_a, cfg_b, gc_resources, "gc-warmup-A", "gc-warmup-B");
        std::printf("  warmup round %d/%d done\n", round + 1, rounds);
    }

    std::printf("Warmup done: Green Context streams.\n");
}

void warmup_green_context_triple(const DeviceCaps& caps,
                                 const LaunchConfig& cfg_a,
                                 const LaunchConfig& cfg_b,
                                 const LaunchConfig& cfg_c,
                                 const std::vector<GcResources>& gc_resources,
                                 int rounds) {
    std::printf("\nWarming up Case 2 (%u-SM GC + two streams on %u-SM GC), rounds=%d...\n",
                gc_resources[0].sm_count, gc_resources[1].sm_count, rounds);

    for (int round = 0; round < rounds; ++round) {
        (void)run_green_context_triple_once(
            caps, cfg_a, cfg_b, cfg_c, gc_resources, "gc-warmup-A", "gc-warmup-B", "gc-warmup-C");
        std::printf("  warmup round %d/%d done\n", round + 1, rounds);
    }

    std::printf("Warmup done: heterogeneous Green Context streams.\n");
}

void warmup_primary_context_big_io(const DeviceCaps& caps,
                                   const LaunchConfig& big_cfg,
                                   const IoLaunchConfig& io_cfg,
                                   int io_kernel_count,
                                   int rounds) {
    std::printf("\nWarming up Case 1 (normal streams: 1 big + %d IO kernels), rounds=%d...\n",
                io_kernel_count, rounds);

    for (int round = 0; round < rounds; ++round) {
        (void)run_primary_context_big_io_once(caps, big_cfg, io_cfg, io_kernel_count);
        std::printf("  warmup round %d/%d done\n", round + 1, rounds);
    }

    std::printf("Warmup done: big + IO on normal streams.\n");
}

void warmup_green_context_big_io(const DeviceCaps& caps,
                                 const LaunchConfig& big_cfg,
                                 const IoLaunchConfig& io_cfg,
                                 int io_kernel_count,
                                 const std::vector<GcResources>& gc_resources,
                                 int rounds) {
    std::printf("\nWarming up Case 2 (%u-SM GC for big + %u-SM GC for %d IO kernels), rounds=%d...\n",
                gc_resources[0].sm_count, gc_resources[1].sm_count, io_kernel_count, rounds);

    for (int round = 0; round < rounds; ++round) {
        (void)run_green_context_big_io_once(caps, big_cfg, io_cfg, io_kernel_count, gc_resources);
        std::printf("  warmup round %d/%d done\n", round + 1, rounds);
    }

    std::printf("Warmup done: big + IO on Green Context streams.\n");
}

int parse_int_arg(int argc, char** argv, const char* name, int default_val, bool positive_only = true) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
            const int v = std::atoi(argv[i + 1]);
            if (positive_only && v <= 0) {
                std::fprintf(stderr, "%s must be positive, got %d\n", name, v);
                std::exit(1);
            }
            return v;
        }
    }
    return default_val;
}

bool has_arg(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}

std::string parse_str_arg(int argc, char** argv, const char* name, const char* default_val) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return default_val;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string experiment = parse_str_arg(argc, argv, "--experiment", "legacy");
    const int work_iters    = parse_int_arg(argc, argv, "--work-iters", kDefaultWorkIters);
    const int warmup_rounds = parse_int_arg(argc, argv, "--warmup-rounds", kDefaultWarmupRounds, false);
    const int measure_rounds= parse_int_arg(argc, argv, "--measure-rounds", kDefaultMeasureRounds, false);
    const int gc_sms_a      = parse_int_arg(argc, argv, "--gc-sms-a", kDefaultGcSms);
    const int gc_sms_b      = parse_int_arg(argc, argv, "--gc-sms-b", kDefaultGcSms);
    const int launch_sms_a  = parse_int_arg(argc, argv, "--launch-sms-a", kDefaultLaunchSms);
    const int launch_sms_b  = parse_int_arg(argc, argv, "--launch-sms-b", kDefaultLaunchSms);
    const int launch_sms_c  = parse_int_arg(argc, argv, "--launch-sms-c", 0, false);
    const int io_kernel_count = parse_int_arg(argc, argv, "--io-kernel-count", kDefaultIoKernelCount);
    const int io_blocks = parse_int_arg(argc, argv, "--io-blocks", kDefaultIoBlocks);
    const int io_iters = parse_int_arg(argc, argv, "--io-iters", kDefaultIoIters);
    const bool enable_third_kernel = launch_sms_c > 0;

    if (experiment != "legacy" && experiment != "big-io") {
        std::fprintf(stderr, "--experiment must be either 'legacy' or 'big-io', got '%s'\n",
                     experiment.c_str());
        return 1;
    }

    if (has_arg(argc, argv, "--io-elements")) {
        std::fprintf(stderr,
                     "--io-elements is intentionally fixed to %llu elements in this test. Use --io-iters to scale IO duration.\n",
                     kDefaultIoElements);
        return 1;
    }

    if (launch_sms_c < 0) {
        std::fprintf(stderr, "--launch-sms-c must be non-negative, got %d\n", launch_sms_c);
        return 1;
    }

    const DeviceCaps caps = query_device_caps(0);

    std::printf("Device: %s (sm_%d%d), SMs=%d\n",
                caps.name, caps.major, caps.minor, caps.sm_count);
    std::printf("clock=%d kHz, shared_mem_per_sm=%d bytes, max_optin_shared_per_block=%d bytes\n",
                caps.clock_khz, caps.shared_mem_per_sm, caps.max_optin_shared_per_block);

    if (gc_sms_a + gc_sms_b > caps.sm_count) {
        std::fprintf(stderr, "ERROR: gc-sms-a(%d) + gc-sms-b(%d) = %d > device SM count(%d)\n",
                     gc_sms_a, gc_sms_b, gc_sms_a + gc_sms_b, caps.sm_count);
        return 1;
    }

    if (experiment == "big-io") {
        const LaunchConfig big_cfg = build_launch_config(caps, work_iters, launch_sms_a);
        IoLaunchConfig io_cfg;
        io_cfg.blocks = io_blocks;
        io_cfg.threads = kThreadsPerBlock;
        io_cfg.num_elements = kDefaultIoElements;
        io_cfg.io_iters = static_cast<unsigned long long>(io_iters);

        std::printf("experiment: big-io\n");
        std::printf("big kernel: blocks=%d, threads=%d, dynamic_smem=%d bytes, work_iters=%d\n",
                    big_cfg.blocks, big_cfg.threads, big_cfg.dynamic_smem_bytes, big_cfg.work_iters);
        std::printf("io kernels: count=%d, blocks=%d, threads=%d, num_elements=%llu (fixed), io_iters=%llu\n",
                    io_kernel_count, io_cfg.blocks, io_cfg.threads, io_cfg.num_elements, io_cfg.io_iters);
        std::printf("green context: GC_A=%d SMs for big kernel, GC_B=%d SMs for IO kernels\n",
                    gc_sms_a, gc_sms_b);
        std::printf("warmup_rounds=%d, measure_rounds=%d\n", warmup_rounds, measure_rounds);
        std::printf("note: big-io experiment only uses --launch-sms-a for the big kernel; --launch-sms-b/c are ignored.\n");

        std::vector<BigIoRoundReport> baseline_rounds;
        baseline_rounds.reserve(measure_rounds);
        warmup_primary_context_big_io(caps, big_cfg, io_cfg, io_kernel_count, warmup_rounds);
        for (int round = 0; round < measure_rounds; ++round) {
            baseline_rounds.push_back(
                run_primary_context_big_io_once(caps, big_cfg, io_cfg, io_kernel_count));
        }
        print_big_io_measurement_summary(
            "Case 1: 1 big compute kernel + multiple IO kernels on normal streams",
            baseline_rounds);

        std::vector<GcResources> gc_resources = create_two_green_contexts(caps.device, gc_sms_a, gc_sms_b);
        std::printf("\nGreen Context partitions: GC_A=%u SMs, GC_B=%u SMs\n",
                    gc_resources[0].sm_count, gc_resources[1].sm_count);

        std::vector<BigIoRoundReport> gc_rounds;
        gc_rounds.reserve(measure_rounds);
        warmup_green_context_big_io(caps, big_cfg, io_cfg, io_kernel_count, gc_resources, warmup_rounds);
        for (int round = 0; round < measure_rounds; ++round) {
            gc_rounds.push_back(
                run_green_context_big_io_once(caps, big_cfg, io_cfg, io_kernel_count, gc_resources));
        }
        print_big_io_measurement_summary(
            "Case 2: big compute kernel on GC_A + IO kernels on GC_B",
            gc_rounds);
        print_big_io_comparison_summary(baseline_rounds, gc_rounds);

        for (GcResources& gc : gc_resources) {
            if (gc.green_ctx) {
                CHECK_CU(cuGreenCtxDestroy(gc.green_ctx));
            }
        }

        return 0;
    }

    const LaunchConfig cfg_a = build_launch_config(caps, work_iters, launch_sms_a);
    const LaunchConfig cfg_b = build_launch_config(caps, work_iters, launch_sms_b);
    const LaunchConfig cfg_c = enable_third_kernel
        ? build_launch_config(caps, work_iters, launch_sms_c)
        : LaunchConfig{};

    std::printf("kernel A: blocks=%d, threads=%d, dynamic_smem=%d bytes, work_iters=%d\n",
                cfg_a.blocks, cfg_a.threads, cfg_a.dynamic_smem_bytes, cfg_a.work_iters);
    std::printf("kernel B: blocks=%d, threads=%d, dynamic_smem=%d bytes, work_iters=%d\n",
                cfg_b.blocks, cfg_b.threads, cfg_b.dynamic_smem_bytes, cfg_b.work_iters);
    if (enable_third_kernel) {
        std::printf("kernel C: blocks=%d, threads=%d, dynamic_smem=%d bytes, work_iters=%d\n",
                    cfg_c.blocks, cfg_c.threads, cfg_c.dynamic_smem_bytes, cfg_c.work_iters);
    }
    std::printf("green context: GC_A=%d SMs, GC_B=%d SMs\n", gc_sms_a, gc_sms_b);
    std::printf("warmup_rounds=%d, measure_rounds=%d\n", warmup_rounds, measure_rounds);

    std::vector<RoundReport> baseline_rounds;
    baseline_rounds.reserve(measure_rounds);
    char baseline_title[160];
    if (enable_third_kernel) {
        warmup_primary_context_triple(caps, cfg_a, cfg_b, cfg_c, warmup_rounds);
        for (int round = 0; round < measure_rounds; ++round) {
            baseline_rounds.push_back(run_primary_context_triple_once(
                caps, cfg_a, cfg_b, cfg_c, "baseline-stream-A", "baseline-stream-B", "baseline-stream-C"));
        }
        std::snprintf(baseline_title, sizeof(baseline_title),
                      "Case 1: A(%d-SM) + B(%d-SM) + C(%d-SM) on normal streams",
                      launch_sms_a, launch_sms_b, launch_sms_c);
    } else {
        warmup_primary_context_pair(caps, cfg_a, cfg_b, warmup_rounds);
        for (int round = 0; round < measure_rounds; ++round) {
            baseline_rounds.push_back(
                run_primary_context_pair_once(caps, cfg_a, cfg_b, "baseline-stream-A", "baseline-stream-B"));
        }
        std::snprintf(baseline_title, sizeof(baseline_title),
                      "Case 1: A(%d-SM) + B(%d-SM) on normal streams", launch_sms_a, launch_sms_b);
    }
    print_measurement_summary(baseline_title, baseline_rounds);

    std::vector<GcResources> gc_resources = create_two_green_contexts(caps.device, gc_sms_a, gc_sms_b);

    std::printf("\nGreen Context partitions: GC_A=%u SMs, GC_B=%u SMs\n",
                gc_resources[0].sm_count, gc_resources[1].sm_count);

    std::vector<RoundReport> gc_rounds;
    gc_rounds.reserve(measure_rounds);
    char gc_title[192];
    if (enable_third_kernel) {
        warmup_green_context_triple(caps, cfg_a, cfg_b, cfg_c, gc_resources, warmup_rounds);
        for (int round = 0; round < measure_rounds; ++round) {
            gc_rounds.push_back(run_green_context_triple_once(
                caps, cfg_a, cfg_b, cfg_c, gc_resources, "gc-stream-A", "gc-stream-B", "gc-stream-C"));
        }
        std::snprintf(gc_title, sizeof(gc_title),
                      "Case 2: A(%d-SM) on %u-SM GC + B(%d-SM) + C(%d-SM) on %u-SM GC",
                      launch_sms_a, gc_resources[0].sm_count,
                      launch_sms_b, launch_sms_c, gc_resources[1].sm_count);
    } else {
        warmup_green_context_pair(caps, cfg_a, cfg_b, gc_resources, warmup_rounds);
        for (int round = 0; round < measure_rounds; ++round) {
            gc_rounds.push_back(
                run_green_context_pair_once(caps, cfg_a, cfg_b, gc_resources, "gc-stream-A", "gc-stream-B"));
        }
        std::snprintf(gc_title, sizeof(gc_title),
                      "Case 2: A(%d-SM) + B(%d-SM) on %u/%u Green Context streams",
                      launch_sms_a, launch_sms_b, gc_resources[0].sm_count, gc_resources[1].sm_count);
    }
    print_measurement_summary(gc_title, gc_rounds);
    print_comparison_summary(baseline_rounds, gc_rounds);

    for (GcResources& gc : gc_resources) {
        if (gc.green_ctx) {
            CHECK_CU(cuGreenCtxDestroy(gc.green_ctx));
        }
    }

    return 0;
}
