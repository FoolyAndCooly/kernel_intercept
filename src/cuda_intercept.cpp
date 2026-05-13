/**
 * @file cuda_intercept.cpp
 * @brief CUDA Runtime API 拦截层实现
 *
 * 本文件实现了 Orion 调度系统的 CUDA API 拦截功能，通过 LD_PRELOAD 机制
 * 拦截应用程序的 CUDA 调用，将其重定向到调度器队列。
 *
 * 工作原理：
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                      LD_PRELOAD 拦截机制                                 │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  1. 应用程序调用 cudaLaunchKernel() 等 CUDA API                         │
 * │  2. 由于 LD_PRELOAD，调用被重定向到本文件中的 wrapper 函数               │
 * │  3. Wrapper 函数创建 OperationRecord，提交到调度器队列                   │
 * │  4. 客户端线程等待调度器执行完成                                         │
 * │  5. 调度器线程调用 execute_cuda_operation() 执行真实的 CUDA 操作         │
 * │  6. 执行完成后唤醒客户端线程，返回结果                                   │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * 拦截的 CUDA API：
 * - cudaMalloc / cudaFree: 内存分配和释放
 * - cudaMemcpy / cudaMemcpyAsync: 内存拷贝
 * - cudaMemset / cudaMemsetAsync: 内存设置
 * - cudaLaunchKernel: Kernel 启动（最关键的拦截点）
 * - cudaDeviceSynchronize / cudaStreamSynchronize: 同步操作
 *
 * 特殊处理：
 * - dlsym 拦截: cuBLAS 使用 dlsym 动态获取函数，需要特殊处理
 * - cudaStreamGetCaptureInfo_v2 拦截: 跳过 CUDA Graph Capture 检查
 *
 * 线程区分：
 * - 客户端线程 (tl_is_scheduler_thread = false): CUDA 调用被拦截到队列
 * - 调度器线程 (tl_is_scheduler_thread = true): CUDA 调用直接执行
 */

#include "gpu_capture.h"
#include <cuda.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

// ============================================================================
// NVTX profiling disabled for CUDA 13 compatibility
// ============================================================================

// ============================================================================
// 线程局部变量引用
// ============================================================================
// 这些变量在 gpu_capture.cpp 中定义，通过 common.h 中的 extern 声明访问。
// 用于区分客户端线程和调度器线程，实现正确的拦截逻辑。

/**
 * @brief 重入保护标志
 *
 * 当调度器线程正在执行某个操作时设置为 true。
 * 防止执行过程中产生的递归拦截（如 cuBLAS 内部调用 cudaLaunchKernel）。
 */
using orion::tl_in_scheduler_execution;

/**
 * @brief 调度器线程标识
 *
 * 调度器线程设置为 true，其 CUDA 调用直接执行不被拦截。
 * 客户端线程保持 false，其 CUDA 调用会被拦截到队列。
 */
using orion::tl_is_scheduler_thread;

// ============================================================================
// Kernel 参数大小缓存
// ============================================================================
// 由于 CUDA Driver API 查询 kernel 参数大小的开销较大，我们使用缓存来优化。
// 但实际上，对于 CUDA Runtime API 启动的 kernel，我们无法直接获取参数大小，
// 因此使用固定大小策略。

/**
 * @brief Kernel 参数大小缓存表
 *
 * Key: kernel 函数指针
 * Value: 参数总大小（字节）
 */
static std::unordered_map<const void*, size_t> g_kernel_param_size_cache;

/**
 * @brief 保护缓存表的互斥锁
 */
static std::mutex g_kernel_cache_mutex;

/**
 * @brief 获取 kernel 参数总大小
 *
 * 由于 CUDA Runtime API 不提供查询 kernel 参数大小的接口，
 * 我们使用保守的固定大小策略：256 字节足以覆盖绝大多数 kernel。
 *
 * 注意：这个函数目前主要用于深拷贝参数的场景，但由于无法准确获取
 * 参数大小，cudaLaunchKernel 的拦截仍然使用同步模式。
 *
 * @param func kernel 函数指针
 * @return 参数总大小（字节）
 */
static size_t get_kernel_param_size(const void* func) {
    // 先查缓存
    {
        std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
        auto it = g_kernel_param_size_cache.find(func);
        if (it != g_kernel_param_size_cache.end()) {
            return it->second;
        }
    }

    // 对于 CUDA Runtime API 启动的 kernel，我们无法直接获取参数大小
    // 使用保守的固定大小策略：256 字节足以覆盖绝大多数 kernel
    size_t param_size = 256;

    // 缓存结果
    {
        std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
        g_kernel_param_size_cache[func] = param_size;
    }

    return param_size;
}

namespace orion {

// ============================================================================
// 真实 CUDA 函数指针类型定义
// ============================================================================
// 定义所有需要拦截的 CUDA Runtime API 的函数指针类型。
// 这些类型用于存储通过 dlsym 获取的真实函数地址。

using cudaMalloc_t = cudaError_t (*)(void**, size_t);
using cudaFree_t = cudaError_t (*)(void*);
using cudaMemcpy_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
using cudaMemcpyAsync_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
using cudaMemset_t = cudaError_t (*)(void*, int, size_t);
using cudaMemsetAsync_t = cudaError_t (*)(void*, int, size_t, cudaStream_t);
using cudaLaunchKernel_t = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
using cudaLaunchKernelExC_t = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
using cuLaunchKernel_t = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                       unsigned, unsigned, unsigned,
                                       unsigned, CUstream, void**, void**);
using cudaDeviceSynchronize_t = cudaError_t (*)();
using cudaStreamSynchronize_t = cudaError_t (*)(cudaStream_t);
using cudaEventSynchronize_t = cudaError_t (*)(cudaEvent_t);

// ============================================================================
// 真实函数指针存储
// ============================================================================
// 存储通过 dlsym 获取的真实 CUDA 函数指针。
// 使用延迟初始化，在第一次调用时获取函数地址。

/**
 * @brief 真实 CUDA 函数指针结构体
 *
 * 包含所有被拦截的 CUDA API 的真实函数指针。
 * initialized 标志确保只初始化一次。
 */
static struct {
    cudaMalloc_t cudaMalloc;
    cudaFree_t cudaFree;
    cudaMemcpy_t cudaMemcpy;
    cudaMemcpyAsync_t cudaMemcpyAsync;
    cudaMemset_t cudaMemset;
    cudaMemsetAsync_t cudaMemsetAsync;
    cudaLaunchKernel_t cudaLaunchKernel;
    cudaLaunchKernelExC_t cudaLaunchKernelExC;
    cuLaunchKernel_t cuLaunchKernel;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaStreamSynchronize_t cudaStreamSynchronize;
    cudaEventSynchronize_t cudaEventSynchronize;
    bool initialized;           // 是否已初始化
    std::mutex init_mutex;      // 保护初始化过程的互斥锁
} g_real_funcs = {nullptr};

/**
 * @brief CUDA Runtime 库句柄
 *
 * 通过 dlopen 打开的 libcudart.so 句柄，用于 dlsym 查找函数。
 */
static void* g_cudart_handle = nullptr;

/**
 * @brief 获取 CUDA 函数指针
 *
 * 通过 dlsym 从 CUDA Runtime 库中获取函数指针。
 * 尝试多个库版本以提高兼容性。
 *
 * 查找顺序：
 * 1. 尝试从已加载的 libcudart.so.12 获取
 * 2. 尝试从已加载的 libcudart.so.11 获取
 * 3. 尝试从已加载的 libcudart.so 获取
 * 4. 如果都失败，尝试加载库后获取
 * 5. 最后尝试从 RTLD_DEFAULT 获取
 *
 * @param name 函数名称
 * @return 函数指针，失败返回 nullptr
 */
static void* get_cuda_func(const char* name) {
    if (!g_cudart_handle) {
        // 尝试多个 CUDA Runtime 库版本
        const char* lib_paths[] = {
            "libcudart.so.12",  // CUDA 12.x
            "libcudart.so.11",  // CUDA 11.x
            "libcudart.so",     // 通用版本
            nullptr
        };

        // 首先尝试从已加载的库中获取（RTLD_NOLOAD）
        for (int i = 0; lib_paths[i]; i++) {
            g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cudart_handle) break;
        }

        // 如果没有已加载的库，尝试加载
        if (!g_cudart_handle) {
            for (int i = 0; lib_paths[i]; i++) {
                g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_cudart_handle) break;
            }
        }
    }

    // 从库句柄中查找函数
    if (g_cudart_handle) {
        void* fn = dlsym(g_cudart_handle, name);
        if (fn) return fn;
    }

    // 最后尝试从默认符号表查找
    return dlsym(RTLD_DEFAULT, name);
}

/**
 * @brief 初始化真实函数指针
 *
 * 获取所有被拦截的 CUDA API 的真实函数指针。
 * 使用互斥锁确保线程安全，只初始化一次。
 */
static void init_real_functions() {
    std::lock_guard<std::mutex> lock(g_real_funcs.init_mutex);
    if (g_real_funcs.initialized) return;

    g_real_funcs.cudaMalloc = (cudaMalloc_t)get_cuda_func("cudaMalloc");
    g_real_funcs.cudaFree = (cudaFree_t)get_cuda_func("cudaFree");
    g_real_funcs.cudaMemcpy = (cudaMemcpy_t)get_cuda_func("cudaMemcpy");
    g_real_funcs.cudaMemcpyAsync = (cudaMemcpyAsync_t)get_cuda_func("cudaMemcpyAsync");
    g_real_funcs.cudaMemset = (cudaMemset_t)get_cuda_func("cudaMemset");
    g_real_funcs.cudaMemsetAsync = (cudaMemsetAsync_t)get_cuda_func("cudaMemsetAsync");
    g_real_funcs.cudaLaunchKernel = (cudaLaunchKernel_t)get_cuda_func("cudaLaunchKernel");
    // Step C：per-thread-stream 变体优先，拿不到再退回普通符号
    g_real_funcs.cudaLaunchKernelExC =
        (cudaLaunchKernelExC_t)get_cuda_func("cudaLaunchKernelExC_ptsz");
    if (!g_real_funcs.cudaLaunchKernelExC) {
        g_real_funcs.cudaLaunchKernelExC =
            (cudaLaunchKernelExC_t)get_cuda_func("cudaLaunchKernelExC");
    }
    g_real_funcs.cuLaunchKernel = (cuLaunchKernel_t)get_cuda_func("cuLaunchKernel");
    g_real_funcs.cudaDeviceSynchronize = (cudaDeviceSynchronize_t)get_cuda_func("cudaDeviceSynchronize");
    g_real_funcs.cudaStreamSynchronize = (cudaStreamSynchronize_t)get_cuda_func("cudaStreamSynchronize");
    g_real_funcs.cudaEventSynchronize = (cudaEventSynchronize_t)get_cuda_func("cudaEventSynchronize");

    g_real_funcs.initialized = true;
    LOG_DEBUG("Real CUDA functions initialized");
}

/**
 * @brief 获取真实函数宏（带延迟初始化）
 *
 * 在调用真实 CUDA 函数前使用此宏确保函数指针已初始化。
 * 如果获取失败，返回 cudaErrorUnknown。
 *
 * @param name 函数名称
 */
#define GET_REAL_FUNC(name) \
    do { \
        if (!g_real_funcs.initialized) init_real_functions(); \
        if (!g_real_funcs.name) { \
            LOG_ERROR("Failed to get real " #name); \
            return cudaErrorUnknown; \
        } \
    } while(0)

/**
 * @brief 安全透传宏
 *
 * 如果调度器未初始化或当前是调度器线程，直接调用真实函数。
 * 这是拦截逻辑的核心判断：
 * - 调度器未初始化：系统还没准备好，直接执行
 * - 调度器线程：避免无限递归，直接执行
 *
 * 关键设计：使用 tl_is_scheduler_thread 而不是 tl_in_scheduler_execution，
 * 这样客户端线程的所有 CUDA 调用（包括通过 cuBLAS 间接调用的）都会被拦截。
 *
 * @param func_name 函数名称
 * @param ... 函数参数
 */
#define SAFE_PASSTHROUGH(func_name, ...) \
    do { \
        if (!g_capture_state.initialized.load() || tl_is_scheduler_thread) { \
            if (!g_real_funcs.initialized) init_real_functions(); \
            if (g_real_funcs.func_name) { \
                return g_real_funcs.func_name(__VA_ARGS__); \
            } \
            return cudaErrorUnknown; \
        } \
    } while(0)

// Step C: driver API 专用透传宏，返回 CUresult
#define SAFE_PASSTHROUGH_DRV(func_name, ...) \
    do { \
        if (!g_capture_state.initialized.load() || tl_is_scheduler_thread) { \
            if (!g_real_funcs.initialized) init_real_functions(); \
            if (g_real_funcs.func_name) { \
                return g_real_funcs.func_name(__VA_ARGS__); \
            } \
            return CUDA_ERROR_UNKNOWN; \
        } \
    } while(0)

// ============================================================================
// 执行真实操作的函数（由调度器调用）
// ============================================================================
// 这些函数在调度器线程中执行，调用真实的 CUDA API。
// 它们从 OperationRecord 中提取参数，执行操作，并返回结果。

/**
 * @brief 执行 cudaMalloc 操作
 *
 * @param op 操作记录（包含 MallocParams）
 * @return CUDA 错误码
 */
cudaError_t execute_malloc(OperationPtr op) {
    GET_REAL_FUNC(cudaMalloc);
    auto& p = std::get<MallocParams>(op->params);
    cudaError_t err = g_real_funcs.cudaMalloc(p.devPtr, p.size);
    op->result_ptr = *p.devPtr;  // 保存分配的指针，供客户端使用
    return err;
}

/**
 * @brief 执行 cudaFree 操作
 *
 * @param op 操作记录（包含 FreeParams）
 * @return CUDA 错误码
 */
cudaError_t execute_free(OperationPtr op) {
    GET_REAL_FUNC(cudaFree);
    auto& p = std::get<FreeParams>(op->params);
    return g_real_funcs.cudaFree(p.devPtr);
}

/**
 * @brief 执行 cudaMemcpy / cudaMemcpyAsync 操作
 *
 * 根据操作类型和调度器 stream 决定使用同步还是异步版本。
 * 如果提供了调度器 stream，总是使用异步版本以避免阻塞。
 *
 * @param op 操作记录（包含 MemcpyParams）
 * @param scheduler_stream 调度器分配的 CUDA stream
 * @return CUDA 错误码
 */
cudaError_t execute_memcpy(OperationPtr op, cudaStream_t scheduler_stream) {
    auto& p = std::get<MemcpyParams>(op->params);
    // 使用调度器的 stream 而不是客户端原始的 stream
    cudaStream_t stream_to_use = scheduler_stream ? scheduler_stream : p.stream;

    if (p.is_async || scheduler_stream) {
        // 如果有调度器 stream，总是使用异步版本
        GET_REAL_FUNC(cudaMemcpyAsync);
        return g_real_funcs.cudaMemcpyAsync(p.dst, p.src, p.count, p.kind, stream_to_use);
    } else {
        GET_REAL_FUNC(cudaMemcpy);
        return g_real_funcs.cudaMemcpy(p.dst, p.src, p.count, p.kind);
    }
}

/**
 * @brief 执行 cudaMemset / cudaMemsetAsync 操作
 *
 * @param op 操作记录（包含 MemsetParams）
 * @param scheduler_stream 调度器分配的 CUDA stream
 * @return CUDA 错误码
 */
cudaError_t execute_memset(OperationPtr op, cudaStream_t scheduler_stream) {
    auto& p = std::get<MemsetParams>(op->params);
    // 使用调度器的 stream 而不是客户端原始的 stream
    cudaStream_t stream_to_use = scheduler_stream ? scheduler_stream : p.stream;

    if (p.is_async || scheduler_stream) {
        // 如果有调度器 stream，总是使用异步版本
        GET_REAL_FUNC(cudaMemsetAsync);
        return g_real_funcs.cudaMemsetAsync(p.devPtr, p.value, p.count, stream_to_use);
    } else {
        GET_REAL_FUNC(cudaMemset);
        return g_real_funcs.cudaMemset(p.devPtr, p.value, p.count);
    }
}

/**
 * @brief 调度器 kernel 执行事件
 *
 * 用于跟踪 kernel 执行状态（预留，当前未使用）。
 */
static cudaEvent_t g_scheduler_event = nullptr;

/**
 * @brief 事件初始化标志
 */
static std::once_flag g_event_init_flag;

/**
 * @brief 执行 cudaLaunchKernel 操作
 *
 * 这是最关键的执行函数，负责在调度器线程中启动 CUDA kernel。
 *
 * 执行流程：
 * 1. 初始化事件（首次调用）
 * 2. 从 OperationRecord 提取 KernelLaunchParams
 * 3. 获取参数指针（原始指针或深拷贝重建的指针）
 * 4. 使用调度器的 stream 启动 kernel
 *
 * @param op 操作记录（包含 KernelLaunchParams）
 * @param scheduler_stream 调度器分配的 CUDA stream
 * @return CUDA 错误码
 */
cudaError_t execute_kernel_launch(OperationPtr op, cudaStream_t scheduler_stream) {
    // 初始化事件（只执行一次）
    std::call_once(g_event_init_flag, []() {
        cudaEventCreate(&g_scheduler_event);
    });

    // Verify parameter type
    if (op->params.index() != 0) {
        LOG_ERROR("Wrong variant index! Expected 0 (KernelLaunchParams), got %zu", op->params.index());
        return cudaErrorUnknown;
    }

    auto& p = std::get<KernelLaunchParams>(op->params);
    void** args = p.get_args();  // 获取参数指针数组

    // 使用调度器分配的 stream 而不是客户端的 stream
    cudaStream_t stream_to_use = scheduler_stream ? scheduler_stream : p.stream;

    // Step C：按拦截来源选择 real_func。Runtime 的 launch（KERNEL_LAUNCH /
    // KERNEL_LAUNCH_EX）用 cudaLaunchKernel；Driver 的 cuLaunchKernel
    // 必须用 cuLaunchKernel，因为 func 是 CUfunction，不是 host entry。
    if (op->type == OperationType::KERNEL_LAUNCH_DRV) {
        if (!g_real_funcs.initialized) init_real_functions();
        if (!g_real_funcs.cuLaunchKernel) {
            LOG_ERROR("cuLaunchKernel symbol unavailable");
            return cudaErrorUnknown;
        }
        CUresult cr = g_real_funcs.cuLaunchKernel(
            reinterpret_cast<CUfunction>(const_cast<void*>(p.func)),
            p.gridDim.x, p.gridDim.y, p.gridDim.z,
            p.blockDim.x, p.blockDim.y, p.blockDim.z,
            static_cast<unsigned>(p.sharedMem),
            reinterpret_cast<CUstream>(stream_to_use),
            args, nullptr
        );
        return cr == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
    }

    GET_REAL_FUNC(cudaLaunchKernel);
    // KERNEL_LAUNCH 与 KERNEL_LAUNCH_EX 都走 cudaLaunchKernel：ExC 路径的
    // attrs（cluster 等）在 Orion 的 ISOLATED / GC 语义下没有额外影响，
    // 归一化到 cudaLaunchKernel 能直接复用现有 real_func。
    cudaError_t result = g_real_funcs.cudaLaunchKernel(
        p.func, p.gridDim, p.blockDim,
        args,
        p.sharedMem, stream_to_use
    );

    return result;
}

/**
 * @brief 执行 cudaDeviceSynchronize 操作
 *
 * @param op 操作记录（包含 SyncParams，但此操作不使用）
 * @return CUDA 错误码
 */
cudaError_t execute_device_sync(OperationPtr op) {
    GET_REAL_FUNC(cudaDeviceSynchronize);
    (void)op;  // 未使用
    return g_real_funcs.cudaDeviceSynchronize();
}

/**
 * @brief 执行 cudaStreamSynchronize 操作
 *
 * @param op 操作记录（包含 SyncParams）
 * @return CUDA 错误码
 */
cudaError_t execute_stream_sync(OperationPtr op) {
    GET_REAL_FUNC(cudaStreamSynchronize);
    auto& p = std::get<SyncParams>(op->params);
    return g_real_funcs.cudaStreamSynchronize(p.stream);
}

/**
 * @brief 执行 CUDA 操作的统一入口（导出给调度器使用）
 *
 * 这是调度器调用的主要接口，根据操作类型分发到具体的执行函数。
 *
 * 执行流程：
 * 1. 设置重入保护标志 (tl_in_scheduler_execution = true)
 * 2. 根据操作类型调用对应的执行函数
 * 3. 清除重入保护标志
 * 4. 返回执行结果
 *
 * 特殊处理：
 * - DEVICE_SYNC: 如果有调度器 stream，只同步该 stream 而不是整个设备
 *   这避免了跨线程死锁问题
 *
 * @param op 操作记录
 * @param scheduler_stream 调度器分配的 CUDA stream
 * @return CUDA 错误码
 */
cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    LOG_DEBUG("execute_cuda_operation: op type=%d, stream=%p", (int)op->type, scheduler_stream);
    
    // 设置重入标志，防止执行过程中的 CUDA 调用被再次拦截
    tl_in_scheduler_execution = true;
    
    cudaError_t result;
    switch (op->type) {
        case OperationType::MALLOC:
            result = execute_malloc(op);
            break;
        case OperationType::FREE:
            result = execute_free(op);
            break;
        case OperationType::MEMCPY:
        case OperationType::MEMCPY_ASYNC:
            result = execute_memcpy(op, scheduler_stream);
            break;
        case OperationType::MEMSET:
        case OperationType::MEMSET_ASYNC:
            result = execute_memset(op, scheduler_stream);
            break;
        case OperationType::KERNEL_LAUNCH:
        case OperationType::KERNEL_LAUNCH_EX:
        case OperationType::KERNEL_LAUNCH_DRV:
            LOG_DEBUG("Entering execute_kernel_launch (type=%s) with stream=%p",
                      op_type_name(op->type), scheduler_stream);
            result = execute_kernel_launch(op, scheduler_stream);
            LOG_DEBUG("execute_kernel_launch returned %d", (int)result);
            break;
        case OperationType::DEVICE_SYNC:
            // 只同步当前调度器的 stream，避免跨线程死锁
            if (scheduler_stream) {
                result = g_real_funcs.cudaStreamSynchronize(scheduler_stream);
            } else {
                result = execute_device_sync(op);
            }
            break;
        case OperationType::STREAM_SYNC:
            result = execute_stream_sync(op);
            break;
        default:
            LOG_ERROR("Unknown operation type for execution: %d", (int)op->type);
            result = cudaErrorUnknown;
            break;
    }
    
    tl_in_scheduler_execution = false;
    return result;
}

// ============================================================================
// 直接调用真实函数的版本（用于非管理线程）
// ============================================================================
// 这些函数直接调用真实的 CUDA API，不经过调度器。
// 用于非管理线程（未注册的线程）或调度器未初始化时。

/**
 * @brief 直接调用真实的 cudaMalloc
 */
cudaError_t real_cudaMalloc(void** devPtr, size_t size) {
    GET_REAL_FUNC(cudaMalloc);
    return g_real_funcs.cudaMalloc(devPtr, size);
}

/**
 * @brief 直接调用真实的 cudaFree
 */
cudaError_t real_cudaFree(void* devPtr) {
    GET_REAL_FUNC(cudaFree);
    return g_real_funcs.cudaFree(devPtr);
}

/**
 * @brief 直接调用真实的 cudaMemcpy
 */
cudaError_t real_cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    GET_REAL_FUNC(cudaMemcpy);
    return g_real_funcs.cudaMemcpy(dst, src, count, kind);
}

/**
 * @brief 直接调用真实的 cudaMemcpyAsync
 */
cudaError_t real_cudaMemcpyAsync(void* dst, const void* src, size_t count,
                                  cudaMemcpyKind kind, cudaStream_t stream) {
    GET_REAL_FUNC(cudaMemcpyAsync);
    return g_real_funcs.cudaMemcpyAsync(dst, src, count, kind, stream);
}

/**
 * @brief 直接调用真实的 cudaMemset
 */
cudaError_t real_cudaMemset(void* devPtr, int value, size_t count) {
    GET_REAL_FUNC(cudaMemset);
    return g_real_funcs.cudaMemset(devPtr, value, count);
}

/**
 * @brief 直接调用真实的 cudaMemsetAsync
 */
cudaError_t real_cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    GET_REAL_FUNC(cudaMemsetAsync);
    return g_real_funcs.cudaMemsetAsync(devPtr, value, count, stream);
}

/**
 * @brief 直接调用真实的 cudaLaunchKernel
 */
cudaError_t real_cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                                   void** args, size_t sharedMem, cudaStream_t stream) {
    GET_REAL_FUNC(cudaLaunchKernel);
    return g_real_funcs.cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

/**
 * @brief 直接调用真实的 cudaDeviceSynchronize
 */
cudaError_t real_cudaDeviceSynchronize() {
    GET_REAL_FUNC(cudaDeviceSynchronize);
    return g_real_funcs.cudaDeviceSynchronize();
}

/**
 * @brief 直接调用真实的 cudaStreamSynchronize
 */
cudaError_t real_cudaStreamSynchronize(cudaStream_t stream) {
    GET_REAL_FUNC(cudaStreamSynchronize);
    return g_real_funcs.cudaStreamSynchronize(stream);
}

} // namespace orion

// ============================================================================
// CUDA API Wrappers (LD_PRELOAD 拦截点)
// ============================================================================
// 这些函数是 LD_PRELOAD 的拦截点，与真实的 CUDA API 同名。
// 当应用程序调用 CUDA API 时，会被重定向到这些 wrapper 函数。
//
// 每个 wrapper 函数的通用流程：
// 1. SAFE_PASSTHROUGH: 检查是否需要透传（调度器未初始化或调度器线程）
// 2. 检查拦截是否启用
// 3. 获取当前线程的客户端索引
// 4. 创建 OperationRecord 并填充参数
// 5. 提交到队列并等待完成
// 6. 返回执行结果

extern "C" {

/**
 * @brief cudaMalloc 拦截 wrapper
 *
 * 拦截 GPU 内存分配请求，提交到调度器队列执行。
 *
 * 对于被管理的线程:
 * 1. 创建 OperationRecord (类型: MALLOC)
 * 2. 填充 MallocParams (devPtr, size)
 * 3. 提交到队列
 * 4. 等待调度器执行完成
 * 5. 返回结果
 *
 * 对于非管理线程:
 * 直接调用真实的 cudaMalloc
 *
 * @param devPtr 输出参数，分配的设备指针
 * @param size 请求分配的大小（字节）
 * @return CUDA 错误码
 */
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMalloc, devPtr, size);
    
    if (!is_capture_enabled()) {
        return real_cudaMalloc(devPtr, size);
    }

    int client_idx = resolve_client_idx();
    if (client_idx < 0) {
        return real_cudaMalloc(devPtr, size);
    }
    
    auto op = create_operation(client_idx, OperationType::MALLOC);
    if (!op) return real_cudaMalloc(devPtr, size);
    
    op->params = MallocParams{devPtr, size};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaFree 拦截 wrapper
 *
 * 拦截 GPU 内存释放请求。
 *
 * @param devPtr 要释放的设备指针
 * @return CUDA 错误码
 */
cudaError_t cudaFree(void* devPtr) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaFree, devPtr);
    
    if (!is_capture_enabled()) return real_cudaFree(devPtr);
    int client_idx = resolve_client_idx();
    if (client_idx < 0) return real_cudaFree(devPtr);
    
    auto op = create_operation(client_idx, OperationType::FREE);
    if (!op) return real_cudaFree(devPtr);
    
    op->params = FreeParams{devPtr};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaMemcpy 拦截 wrapper
 *
 * 拦截同步内存拷贝请求。
 * cudaMemcpy 有隐式同步语义（对于 D2H 和 H2D），通过队列调度来维持这个语义。
 *
 * @param dst 目标地址
 * @param src 源地址
 * @param count 拷贝大小（字节）
 * @param kind 拷贝方向 (H2D, D2H, D2D, H2H)
 * @return CUDA 错误码
 */
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemcpy, dst, src, count, kind);
    
    if (!is_capture_enabled()) return real_cudaMemcpy(dst, src, count, kind);
    int client_idx = resolve_client_idx();
    if (client_idx < 0) return real_cudaMemcpy(dst, src, count, kind);
    
    auto op = create_operation(client_idx, OperationType::MEMCPY);
    if (!op) return real_cudaMemcpy(dst, src, count, kind);
    
    op->params = MemcpyParams{dst, src, count, kind, nullptr, false};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaMemcpyAsync 拦截 wrapper
 *
 * 拦截异步内存拷贝请求。
 *
 * @param dst 目标地址
 * @param src 源地址
 * @param count 拷贝大小（字节）
 * @param kind 拷贝方向
 * @param stream CUDA stream
 * @return CUDA 错误码
 */
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, 
                            cudaMemcpyKind kind, cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemcpyAsync, dst, src, count, kind, stream);
    
    if (!is_capture_enabled()) return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    int client_idx = resolve_client_idx(stream);
    if (client_idx < 0) return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    
    auto op = create_operation(client_idx, OperationType::MEMCPY_ASYNC);
    if (!op) return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    
    op->params = MemcpyParams{dst, src, count, kind, stream, true};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaMemset 拦截 wrapper
 *
 * 拦截同步内存设置请求。
 *
 * @param devPtr 设备指针
 * @param value 设置的值（每字节）
 * @param count 设置大小（字节）
 * @return CUDA 错误码
 */
cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemset, devPtr, value, count);
    
    if (!is_capture_enabled()) return real_cudaMemset(devPtr, value, count);
    int client_idx = resolve_client_idx();
    if (client_idx < 0) return real_cudaMemset(devPtr, value, count);
    
    auto op = create_operation(client_idx, OperationType::MEMSET);
    if (!op) return real_cudaMemset(devPtr, value, count);
    
    op->params = MemsetParams{devPtr, value, count, nullptr, false};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaMemsetAsync 拦截 wrapper
 *
 * 拦截异步内存设置请求。
 *
 * @param devPtr 设备指针
 * @param value 设置的值（每字节）
 * @param count 设置大小（字节）
 * @param stream CUDA stream
 * @return CUDA 错误码
 */
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemsetAsync, devPtr, value, count, stream);
    
    if (!is_capture_enabled()) return real_cudaMemsetAsync(devPtr, value, count, stream);
    int client_idx = resolve_client_idx(stream);
    if (client_idx < 0) return real_cudaMemsetAsync(devPtr, value, count, stream);
    
    auto op = create_operation(client_idx, OperationType::MEMSET_ASYNC);
    if (!op) return real_cudaMemsetAsync(devPtr, value, count, stream);
    
    op->params = MemsetParams{devPtr, value, count, stream, true};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaLaunchKernel 拦截 wrapper
 *
 * 这是最关键的拦截点，所有 CUDA kernel 最终都通过这里发起。
 * PyTorch、cuBLAS、cuDNN 等库最终都会调用 cudaLaunchKernel。
 *
 * 实现调度器执行的关键流程：
 * 1. 检查是否需要透传（调度器线程或未初始化）
 * 2. 创建 OperationRecord (类型: KERNEL_LAUNCH)
 * 3. 填充 KernelLaunchParams (func, gridDim, blockDim, args, sharedMem, stream)
 * 4. 提交到调度器队列
 * 5. 客户端线程等待调度器执行完成
 * 6. 调度器线程调用真实的 cudaLaunchKernel
 *
 * 参数处理说明：
 * - 由于无法获取 kernel 参数的实际大小，目前使用原始指针模式
 * - 客户端线程必须等待操作完成，确保参数在执行时仍然有效
 * - 异步模式主要用于 cuBLAS 等参数已知的高级 API
 *
 * @param func kernel 函数指针
 * @param gridDim Grid 维度 (block 数量)
 * @param blockDim Block 维度 (thread 数量)
 * @param args 参数指针数组
 * @param sharedMem 动态共享内存大小
 * @param stream CUDA stream
 * @return CUDA 错误码
 */
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    using namespace orion;

    SAFE_PASSTHROUGH(cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);

    if (!is_capture_enabled()) return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    int client_idx = resolve_client_idx(stream);
    if (client_idx < 0) return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    // 使用新接口避免竞态条件
    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH);
    if (!op) {
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    // 先设置 params
    KernelLaunchParams kp;
    kp.func = func;
    kp.gridDim = gridDim;
    kp.blockDim = blockDim;
    kp.sharedMem = sharedMem;
    kp.stream = stream;

    // 注意：异步模式目前只对 cudaLaunchKernel 有效
    // 由于无法获取 kernel 参数的实际大小，我们无法安全地深拷贝参数
    // 因此，即使在异步模式下，cudaLaunchKernel 也需要等待操作完成
    //
    // 异步模式的主要优化是针对 cuBLAS 等高级 API，它们的参数是已知的
    // 对于原始的 cudaLaunchKernel，我们保持同步行为

    kp.original_args = args;
    kp.use_deep_copy = false;

    op->params = std::move(kp);
    enqueue_operation(op);

    // 同步模式：等待调度器执行完成
    wait_operation(op);

    return op->result;
}

// ============================================================================
// Step C：cudaLaunchKernelExC / cudaLaunchKernelEx 拦截
// ============================================================================
// PyTorch / CUDA Graph / cooperative launch 会走 cudaLaunchKernelEx 路径，
// 它通过 cudaLaunchConfig_t 带 gridDim/blockDim/dynamicSmemBytes/stream/attrs。
// attrs（cluster 等）对 Orion 调度语义无关，这里忽略；但 stream 和三维
// 维度必须准确保留。
// ----------------------------------------------------------------------------
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                const void* func, void** args) {
    using namespace orion;

    SAFE_PASSTHROUGH(cudaLaunchKernelExC, config, func, args);

    if (!is_capture_enabled() || config == nullptr) {
        GET_REAL_FUNC(cudaLaunchKernelExC);
        return g_real_funcs.cudaLaunchKernelExC(config, func, args);
    }

    int client_idx = resolve_client_idx(config->stream);
    if (client_idx < 0) {
        GET_REAL_FUNC(cudaLaunchKernelExC);
        return g_real_funcs.cudaLaunchKernelExC(config, func, args);
    }

    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH_EX);
    if (!op) {
        GET_REAL_FUNC(cudaLaunchKernelExC);
        return g_real_funcs.cudaLaunchKernelExC(config, func, args);
    }

    KernelLaunchParams kp;
    kp.func          = func;
    kp.gridDim       = config->gridDim;
    kp.blockDim      = config->blockDim;
    kp.sharedMem     = config->dynamicSmemBytes;
    kp.stream        = config->stream;
    kp.original_args = args;
    kp.use_deep_copy = false;

    op->params = std::move(kp);
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

// ============================================================================
// Step C：cuLaunchKernel driver API 拦截
// ============================================================================
// Triton / NCCL / cuGraph 大量走 driver API。参数顺序：
//   cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sharedMem, stream,
//                  kernelParams, extra)
// extra 极少用到（只有少数 runtime fallback 场景），这里透传但不参与深拷贝。
// ----------------------------------------------------------------------------
CUresult cuLaunchKernel(CUfunction f,
                        unsigned gridDimX, unsigned gridDimY, unsigned gridDimZ,
                        unsigned blockDimX, unsigned blockDimY, unsigned blockDimZ,
                        unsigned sharedMemBytes, CUstream hStream,
                        void** kernelParams, void** extra) {
    using namespace orion;

    SAFE_PASSTHROUGH_DRV(cuLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
                         blockDimX, blockDimY, blockDimZ,
                         sharedMemBytes, hStream, kernelParams, extra);

    if (!is_capture_enabled()) {
        if (!g_real_funcs.initialized) init_real_functions();
        if (!g_real_funcs.cuLaunchKernel) return CUDA_ERROR_UNKNOWN;
        return g_real_funcs.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ,
                                           sharedMemBytes, hStream,
                                           kernelParams, extra);
    }

    // driver stream 和 runtime stream 在当前 CUDA 版本中 ABI 一致，
    // resolve_client_idx 直接按指针匹配即可。
    cudaStream_t rt_stream = reinterpret_cast<cudaStream_t>(hStream);
    int client_idx = resolve_client_idx(rt_stream);
    if (client_idx < 0 || extra != nullptr) {
        // extra 参数罕见，且我们没有把它存进 KernelLaunchParams，遇到时
        // 不走队列，直接透传，避免丢信息。
        if (!g_real_funcs.initialized) init_real_functions();
        if (!g_real_funcs.cuLaunchKernel) return CUDA_ERROR_UNKNOWN;
        return g_real_funcs.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ,
                                           sharedMemBytes, hStream,
                                           kernelParams, extra);
    }

    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH_DRV);
    if (!op) {
        if (!g_real_funcs.initialized) init_real_functions();
        if (!g_real_funcs.cuLaunchKernel) return CUDA_ERROR_UNKNOWN;
        return g_real_funcs.cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ,
                                           sharedMemBytes, hStream,
                                           kernelParams, extra);
    }

    KernelLaunchParams kp;
    kp.func          = reinterpret_cast<const void*>(f);
    kp.gridDim       = dim3(gridDimX, gridDimY, gridDimZ);
    kp.blockDim      = dim3(blockDimX, blockDimY, blockDimZ);
    kp.sharedMem     = sharedMemBytes;
    kp.stream        = rt_stream;
    kp.original_args = kernelParams;
    kp.use_deep_copy = false;

    op->params = std::move(kp);
    enqueue_operation(op);
    wait_operation(op);
    // 把 runtime 错误码翻回 driver 错误码：成功走 CUDA_SUCCESS，其余一律
    // ERROR_UNKNOWN（调度器已经在执行端用 cuLaunchKernel，就按它返回）。
    return op->result == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

 *
 * 拦截设备级同步操作。这是显式同步操作，需要等待所有之前的操作完成。
 *
 * 特殊处理：
 * - 在异步模式下，通过提交一个同步操作并等待它完成来实现
 * - 这确保了队列中所有之前的操作都已完成
 *
 * @return CUDA 错误码
 */
cudaError_t cudaDeviceSynchronize(void) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaDeviceSynchronize);
    
    if (!is_capture_enabled()) return real_cudaDeviceSynchronize();
    int client_idx = resolve_client_idx();
    if (client_idx < 0) return real_cudaDeviceSynchronize();
    
    // 在异步模式下，需要等待队列中所有操作完成
    // 通过提交一个同步操作并等待它完成来实现
    auto op = create_operation(client_idx, OperationType::DEVICE_SYNC);
    if (!op) return real_cudaDeviceSynchronize();
    
    op->params = SyncParams{nullptr, nullptr};
    enqueue_operation(op);
    
    // 无论是否异步模式，同步操作都需要等待完成
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaStreamSynchronize 拦截 wrapper
 *
 * 拦截 stream 级同步操作。
 *
 * @param stream 要同步的 CUDA stream
 * @return CUDA 错误码
 */
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaStreamSynchronize, stream);
    
    if (!is_capture_enabled()) return real_cudaStreamSynchronize(stream);
    int client_idx = resolve_client_idx(stream);
    if (client_idx < 0) return real_cudaStreamSynchronize(stream);
    
    auto op = create_operation(client_idx, OperationType::STREAM_SYNC);
    if (!op) return real_cudaStreamSynchronize(stream);
    
    op->params = SyncParams{stream, nullptr};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * @brief cudaStreamGetCaptureInfo_v2 拦截
 *
 * cuBLAS 在每次操作前调用此函数检查 stream 是否处于 CUDA Graph Capture 模式。
 * 由于 Orion 不使用 CUDA Graph Capture，直接返回"不在 capture 模式"可以跳过这个检查，
 * 减少约 1.5x cudaLaunchKernel 次数的额外开销。
 *
 * 注意：此拦截可能不会被调用，因为 cuBLAS 可能在 CUDA 驱动层内部调用此函数。
 * 保留此实现以备将来使用。
 *
 * @param stream CUDA stream（忽略）
 * @param captureStatus_out 输出：capture 状态
 * @param id_out 输出：capture ID
 * @param graph_out 输出：graph 句柄
 * @param dependencies_out 输出：依赖节点
 * @param numDependencies_out 输出：依赖数量
 * @return cudaSuccess
 */
cudaError_t cudaStreamGetCaptureInfo_v2(
    cudaStream_t stream,
    cudaStreamCaptureStatus* captureStatus_out,
    unsigned long long* id_out,
    cudaGraph_t* graph_out,
    const cudaGraphNode_t** dependencies_out,
    size_t* numDependencies_out) {
    
    (void)stream;  // 忽略 stream 参数
    
    // 注意：此拦截可能不会被调用，因为 cuBLAS 可能在 CUDA 驱动层内部调用此函数
    // 保留此实现以备将来使用
    
    // 直接返回"不在 capture 模式"
    if (captureStatus_out) *captureStatus_out = cudaStreamCaptureStatusNone;
    if (id_out) *id_out = 0;
    if (graph_out) *graph_out = nullptr;
    if (dependencies_out) *dependencies_out = nullptr;
    if (numDependencies_out) *numDependencies_out = 0;
    
    return cudaSuccess;
}

/**
 * @brief cudaStreamIsCapturing 拦截（旧版 API）
 *
 * 与 cudaStreamGetCaptureInfo_v2 类似，用于检查 stream 是否在 capture 模式。
 * 直接返回"不在 capture 模式"。
 *
 * @param stream CUDA stream（忽略）
 * @param pCaptureStatus 输出：capture 状态
 * @return cudaSuccess
 */
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
    (void)stream;
    if (pCaptureStatus) *pCaptureStatus = cudaStreamCaptureStatusNone;
    return cudaSuccess;
}

/**
 * @brief dlsym 拦截
 *
 * cuBLAS 使用 dlsym 动态获取 cudaStreamGetCaptureInfo_v2 等函数，
 * 绕过了 LD_PRELOAD 的符号拦截。我们通过拦截 dlsym 本身来解决这个问题。
 *
 * 拦截的符号：
 * - cudaStreamGetCaptureInfo_v2: 返回我们的拦截版本
 * - cudaStreamIsCapturing: 返回我们的拦截版本
 *
 * 其他符号使用真实的 dlsym 查找。
 *
 * @param handle 库句柄
 * @param symbol 符号名称
 * @return 符号地址
 */
static void* (*real_dlsym)(void*, const char*) = nullptr;

void* dlsym(void* handle, const char* symbol) {
    // 获取真实的 dlsym（使用 dlvsym 避免递归）
    if (!real_dlsym) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            // 如果找不到特定版本，尝试默认版本
            real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", nullptr);
        }
    }

    // 拦截 cudaStreamGetCaptureInfo_v2
    if (symbol && strcmp(symbol, "cudaStreamGetCaptureInfo_v2") == 0) {
        return (void*)cudaStreamGetCaptureInfo_v2;
    }

    // 拦截 cudaStreamIsCapturing
    if (symbol && strcmp(symbol, "cudaStreamIsCapturing") == 0) {
        return (void*)cudaStreamIsCapturing;
    }

    // 其他符号使用真实的 dlsym
    if (real_dlsym) {
        return real_dlsym(handle, symbol);
    }
    return nullptr;
}

// ============================================================================
// 异步模式 C 接口
// ============================================================================

/**
 * @brief 设置异步模式（C 接口）
 *
 * @param mode 0=同步模式, 1=异步模式
 */
void orion_set_async_mode(int mode) {
    orion::set_async_mode_internal(mode);
    LOG_INFO("Async mode set to %d (%s)", mode, mode == 0 ? "SYNC" : "ASYNC_KERNEL");
}

/**
 * @brief 获取当前异步模式（C 接口）
 *
 * @return 0=同步模式, 1=异步模式
 */
int orion_get_async_mode() {
    return orion::get_async_mode_internal();
}

} // extern "C"
