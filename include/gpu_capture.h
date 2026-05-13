/**
 * @file gpu_capture.h
 * @brief Orion GPU 操作捕获层头文件（简化版）
 *
 * 架构：
 * - 客户端线程提交操作到队列，然后忙等完成
 * - 调度器线程轮询队列执行操作
 * - 每个队列有互斥锁保护
 */

#ifndef ORION_GPU_CAPTURE_H
#define ORION_GPU_CAPTURE_H

#include "common.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <mutex>
#include <queue>
#include <vector>
#include <memory>
#include <variant>

namespace orion {

// ============================================================================
// 操作参数结构体
// ============================================================================

constexpr size_t MAX_KERNEL_ARGS_SIZE = 4096;
constexpr size_t MAX_KERNEL_ARGS_COUNT = 64;

/**
 * @brief cudaLaunchKernel 参数结构体
 */
struct KernelLaunchParams {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    void** original_args;

    // 深拷贝参数（同步模式下使用 original_args 即可）
    std::vector<uint8_t> args_buffer;
    std::vector<size_t> args_offsets;
    std::vector<size_t> args_sizes;
    size_t num_args;
    std::vector<void*> args_ptrs;
    bool use_deep_copy;

    KernelLaunchParams()
        : func(nullptr), sharedMem(0), stream(nullptr)
        , original_args(nullptr), num_args(0), use_deep_copy(false) {
        gridDim = {1, 1, 1};
        blockDim = {1, 1, 1};
    }

    void** get_args() {
        if (use_deep_copy && num_args > 0) {
            args_ptrs.resize(num_args);
            for (size_t i = 0; i < num_args; i++) {
                args_ptrs[i] = args_buffer.data() + args_offsets[i];
            }
            return args_ptrs.data();
        }
        return original_args;
    }
};

struct MallocParams {
    void** devPtr;
    size_t size;
};

struct FreeParams {
    void* devPtr;
};

struct MemcpyParams {
    void* dst;
    const void* src;
    size_t count;
    cudaMemcpyKind kind;
    cudaStream_t stream;
    bool is_async;
};

struct MemsetParams {
    void* devPtr;
    int value;
    size_t count;
    cudaStream_t stream;
    bool is_async;
};

struct SyncParams {
    cudaStream_t stream;
    cudaEvent_t event;
};

struct CudnnConvParams {
    void* handle;
    void* xDesc;
    const void* x;
    void* wDesc;
    const void* w;
    void* convDesc;
    int algo;
    void* workSpace;
    size_t workSpaceSizeInBytes;
    const void* alpha;
    const void* beta;
    void* yDesc;
    void* y;
};

struct CudnnBatchNormParams {
    void* handle;
    int mode;
    const void* alpha;
    const void* beta;
    void* xDesc;
    const void* x;
    void* yDesc;
    void* y;
    void* bnScaleBiasMeanVarDesc;
    const void* bnScale;
    const void* bnBias;
    double exponentialAverageFactor;
    void* resultRunningMean;
    void* resultRunningVariance;
    double epsilon;
    void* resultSaveMean;
    void* resultSaveInvVariance;
};

struct CublasGemmParams {
    void* handle;
    int transa;
    int transb;
    int m, n, k;
    const void* alpha;
    const void* A;
    int lda;
    const void* B;
    int ldb;
    const void* beta;
    void* C;
    int ldc;
    int batchCount;
    long long strideA, strideB, strideC;
    bool is_batched;
    bool is_strided;
    float alpha_value;
    float beta_value;
    bool use_stored_scalars;

    CublasGemmParams() : handle(nullptr), transa(0), transb(0),
                         m(0), n(0), k(0), alpha(nullptr), A(nullptr), lda(0),
                         B(nullptr), ldb(0), beta(nullptr), C(nullptr), ldc(0),
                         batchCount(0), strideA(0), strideB(0), strideC(0),
                         is_batched(false), is_strided(false),
                         alpha_value(1.0f), beta_value(0.0f), use_stored_scalars(false) {}
};

struct CublasLtMatmulParams {
    void* lightHandle;
    void* computeDesc;
    const void* alpha;
    const void* A;
    void* Adesc;
    const void* B;
    void* Bdesc;
    const void* beta;
    const void* C;
    void* Cdesc;
    void* D;
    void* Ddesc;
    const void* algo;
    void* workspace;
    size_t workspaceSizeInBytes;
    cudaStream_t stream;
    float alpha_value;
    float beta_value;
    bool use_stored_scalars;

    CublasLtMatmulParams() : lightHandle(nullptr), computeDesc(nullptr),
                             alpha(nullptr), A(nullptr), Adesc(nullptr),
                             B(nullptr), Bdesc(nullptr), beta(nullptr),
                             C(nullptr), Cdesc(nullptr), D(nullptr), Ddesc(nullptr),
                             algo(nullptr), workspace(nullptr), workspaceSizeInBytes(0),
                             stream(nullptr), alpha_value(1.0f), beta_value(0.0f),
                             use_stored_scalars(false) {}
};

// ============================================================================
// OperationRecord（简化版：忙等代替条件变量）
// ============================================================================

struct OperationRecord {
    OperationType type;
    uint64_t op_id;
    int client_idx;

    std::variant<
        KernelLaunchParams,
        MallocParams,
        FreeParams,
        MemcpyParams,
        MemsetParams,
        SyncParams,
        CudnnConvParams,
        CudnnBatchNormParams,
        CublasGemmParams,
        CublasLtMatmulParams
    > params;

    // 执行状态
    std::atomic<bool> completed{false};
    std::atomic<bool> started{false};
    cudaError_t result{cudaSuccess};
    void* result_ptr{nullptr};

    OperationRecord() : type(OperationType::UNKNOWN), op_id(0), client_idx(-1),
                        params(MallocParams{}) {}

    ~OperationRecord() = default;

    // 禁止拷贝和移动
    OperationRecord(const OperationRecord&) = delete;
    OperationRecord& operator=(const OperationRecord&) = delete;
    OperationRecord(OperationRecord&&) = delete;
    OperationRecord& operator=(OperationRecord&&) = delete;

    // 忙等完成
    void wait_completion() {
        while (!completed.load(std::memory_order_acquire)) {
            // 纯忙等，最小化延迟
        }
    }

    // 标记完成
    void mark_completed(cudaError_t res) {
        result = res;
        completed.store(true, std::memory_order_release);
    }
};

using OperationPtr = std::shared_ptr<OperationRecord>;

// ============================================================================
// ClientQueue（简化版）
// ============================================================================

class ClientQueue {
public:
    ClientQueue() : shutdown_(false) {}

    void push(OperationPtr op) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(op));
    }

    OperationPtr try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return nullptr;
        OperationPtr op = std::move(queue_.front());
        queue_.pop();
        return op;
    }

    OperationPtr peek() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return nullptr;
        return queue_.front();
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void shutdown() {
        shutdown_.store(true);
    }

    bool is_shutdown() const {
        return shutdown_.load();
    }

private:
    std::queue<OperationPtr> queue_;
    std::mutex mutex_;
    std::atomic<bool> shutdown_;
};

// ============================================================================
// CaptureLayerState（简化版）
// ============================================================================

struct CaptureLayerState {
    std::atomic<bool> initialized{false};
    std::atomic<bool> enabled{false};
    int num_clients{0};

    std::vector<std::unique_ptr<ClientQueue>> client_queues;
    std::atomic<uint64_t> next_op_id{0};
    std::atomic<bool> shutdown{false};
};

extern CaptureLayerState g_capture_state;

// ============================================================================
// API 函数
// ============================================================================

int init_capture_layer(int num_clients);
void shutdown_capture_layer();

int get_current_client_idx();
void set_current_client_idx(int idx);

OperationPtr create_operation(int client_idx, OperationType type);
void enqueue_operation(OperationPtr op);
void wait_operation(OperationPtr op);

// 兼容旧接口
OperationPtr submit_operation(int client_idx, OperationType type);

bool is_managed_thread();
bool is_capture_enabled();
void set_capture_enabled(bool enabled);

// Stream / handle → client_idx 映射（Step B）
// 客户端线程通过 set_current_client_idx 走 thread-local 快路径；
// 对于 cuDNN/cuBLAS 内部 worker 线程，按 stream 或 handle 反查 client。
void register_client_stream(int client_idx, void* stream);
void register_client_handle(int client_idx, void* handle);
int  resolve_client_idx(void* stream = nullptr, void* handle = nullptr);
void clear_client_maps();

// 异步模式控制
void set_async_mode_internal(int mode);
int get_async_mode_internal();

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_init(int num_clients);
void orion_shutdown();
void orion_set_client_idx(int idx);
int orion_get_client_idx();
void orion_set_enabled(int enabled);
int orion_is_enabled();

// Step A 别名 + Step B 新接口
int  orion_set_capture(int enabled);
void orion_register_client_stream(int client_idx, void* stream);
void orion_register_client_handle(int client_idx, void* handle);
int  orion_resolve_client_idx(void* stream, void* handle);

}

#endif // ORION_GPU_CAPTURE_H
