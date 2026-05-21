/**
 * @file cublas_intercept.cpp
 * @brief cuBLAS API 拦截层实现
 *
 * 本文件实现了 Orion 调度系统的 cuBLAS API 拦截功能。
 * cuBLAS 是 NVIDIA 的 GPU 加速线性代数库，PyTorch 等深度学习框架
 * 大量使用 cuBLAS 进行矩阵运算。
 *
 * 拦截的 cuBLAS API：
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  cuBLAS 函数                    │  用途                                 │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  cublasSgemm_v2                 │  单精度矩阵乘法 (FP32)                │
 * │  cublasSgemmBatched             │  批量单精度矩阵乘法                   │
 * │  cublasSgemmStridedBatched      │  跨步批量单精度矩阵乘法               │
 * │  cublasHgemm                    │  半精度矩阵乘法 (FP16)                │
 * │  cublasGemmEx                   │  混合精度矩阵乘法                     │
 * │  cublasGemmStridedBatchedEx     │  混合精度批量矩阵乘法                 │
 * │  cublasHgemmStridedBatched      │  FP16 批量矩阵乘法                    │
 * │  cublasSgemmEx                  │  扩展 FP32 矩阵乘法                   │
 * │  cublasSetStream_v2             │  设置 cuBLAS handle 的 stream         │
 * │  cublasLtMatmul                 │  cuBLASLt 矩阵乘法                    │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * 关键设计：
 * 1. 线程本地 cuBLAS handle: 每个调度器线程有独立的 handle，避免竞争
 * 2. Stream 控制: 拦截 cublasSetStream_v2，让调度器控制执行 stream
 * 3. 异步模式支持: 存储 alpha/beta 标量值，支持异步执行
 * 4. 重入保护: 防止 cuBLAS 内部调用被递归拦截
 *
 * 工作流程：
 * 1. 应用程序调用 cuBLAS API (如 cublasSgemm_v2)
 * 2. LD_PRELOAD 将调用重定向到本文件的 wrapper 函数
 * 3. Wrapper 创建 OperationRecord，填充 CublasGemmParams
 * 4. 提交到调度器队列
 * 5. 调度器线程调用 execute_cublas_operation() 执行
 * 6. 使用线程本地 handle 和调度器 stream 执行真实操作
 */

#include "gpu_capture.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <mutex>

// ============================================================================
// 线程局部变量引用
// ============================================================================

/**
 * @brief 重入保护标志
 *
 * 在 gpu_capture.cpp 中定义，通过 common.h 中的 extern 声明访问。
 * 用于防止 cuBLAS 内部的 cudaLaunchKernel 调用被递归拦截。
 */
using orion::tl_in_scheduler_execution;

// ============================================================================
// ============================================================================
// cuBLAS 类型定义
// ============================================================================
// 避免在此文件中引入 cuBLAS 头文件依赖，手动定义所需类型。

typedef void* cublasHandle_t;       // cuBLAS 句柄类型
typedef int cublasStatus_t;         // cuBLAS 状态/错误码类型
typedef int cublasOperation_t;      // 矩阵转置操作类型

#define CUBLAS_STATUS_SUCCESS 0     // 成功状态码
#define CUBLAS_STATUS_INTERNAL_ERROR 6  // 内部错误
#define CUBLAS_OP_N 0               // 不转置
#define CUBLAS_OP_T 1               // 转置

namespace orion {

// ============================================================================
// cuBLAS 真实函数指针类型
// ============================================================================

using cublasSgemm_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int);

using cublasSgemmBatched_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float* const*, int,
    const float* const*, int,
    const float*, float* const*, int,
    int);

using cublasSgemmStridedBatched_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int, long long,
    const float*, int, long long,
    const float*, float*, int, long long,
    int);

using cublasHgemm_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int,
    const void*, int,
    const void*, void*, int);

using cublasGemmEx_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int);

// cublasGemmStridedBatchedEx - 混合精度批量 GEMM
using cublasGemmStridedBatchedEx_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*,
    const void*, int, int, long long,
    const void*, int, int, long long,
    const void*,
    void*, int, int, long long,
    int,
    int, int);

// cublasHgemmStridedBatched - FP16 批量 GEMM
using cublasHgemmStridedBatched_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int, long long,
    const void*, int, long long,
    const void*, void*, int, long long,
    int);

// cublasSgemmEx - 扩展 FP32 GEMM
using cublasSgemmEx_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*,
    const void*, int, int,
    const void*, int, int,
    const float*,
    void*, int, int);

// cublasSetStream - 设置 cuBLAS handle 的 stream
using cublasSetStream_t = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
using cublasGetStream_t = cublasStatus_t (*)(cublasHandle_t, cudaStream_t*);

} // namespace orion

// ============================================================================
// cuBLASLt 类型定义 (全局作用域，避免包含 cublasLt.h)
// ============================================================================

typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulAlgo_t;

// cublasLtMatmul 函数签名
typedef cublasStatus_t (*cublasLtMatmul_t)(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void* alpha,
    const void* A,
    cublasLtMatrixLayout_t Adesc,
    const void* B,
    cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C,
    cublasLtMatrixLayout_t Cdesc,
    void* D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream);

namespace orion {

// ============================================================================
// 真实函数指针存储
// ============================================================================

static struct {
    cublasSgemm_t cublasSgemm_v2;
    cublasSgemmBatched_t cublasSgemmBatched;
    cublasSgemmStridedBatched_t cublasSgemmStridedBatched;
    cublasHgemm_t cublasHgemm;
    cublasGemmEx_t cublasGemmEx;
    cublasGemmStridedBatchedEx_t cublasGemmStridedBatchedEx;
    cublasHgemmStridedBatched_t cublasHgemmStridedBatched;
    cublasSgemmEx_t cublasSgemmEx;
    cublasSetStream_t cublasSetStream_v2;
    cublasGetStream_t cublasGetStream_v2;
    bool initialized;
    std::mutex init_mutex;
} g_cublas_funcs = {nullptr};

static void* g_cublas_handle = nullptr;

// cuBLASLt 函数指针存储
static struct {
    cublasLtMatmul_t cublasLtMatmul;
    bool initialized;
    std::mutex init_mutex;
} g_cublaslt_funcs = {nullptr};

static void* g_cublaslt_handle = nullptr;

static void* get_cublas_func(const char* name) {
    // 如果还没有打开 libcublas，尝试打开它
    if (!g_cublas_handle) {
        // 尝试多个可能的库路径
        const char* lib_paths[] = {
            "libcublas.so.12",
            "libcublas.so.11", 
            "libcublas.so",
            nullptr
        };
        
        for (int i = 0; lib_paths[i]; i++) {
            // 使用 RTLD_NOLOAD 获取已加载的库句柄
            g_cublas_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cublas_handle) {
                LOG_DEBUG("Found cuBLAS library: %s", lib_paths[i]);
                break;
            }
        }
        
        // 如果 RTLD_NOLOAD 失败，尝试正常加载
        if (!g_cublas_handle) {
            for (int i = 0; lib_paths[i]; i++) {
                g_cublas_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_cublas_handle) {
                    LOG_DEBUG("Loaded cuBLAS library: %s", lib_paths[i]);
                    break;
                }
            }
        }
    }
    
    if (g_cublas_handle) {
        void* fn = dlsym(g_cublas_handle, name);
        if (fn) return fn;
    }
    
    // 备选：使用 RTLD_DEFAULT
    return dlsym(RTLD_DEFAULT, name);
}

static void init_cublas_functions() {
    std::lock_guard<std::mutex> lock(g_cublas_funcs.init_mutex);
    if (g_cublas_funcs.initialized) return;
    
    g_cublas_funcs.cublasSgemm_v2 = 
        (cublasSgemm_t)get_cublas_func("cublasSgemm_v2");
    g_cublas_funcs.cublasSgemmBatched = 
        (cublasSgemmBatched_t)get_cublas_func("cublasSgemmBatched");
    g_cublas_funcs.cublasSgemmStridedBatched = 
        (cublasSgemmStridedBatched_t)get_cublas_func("cublasSgemmStridedBatched");
    g_cublas_funcs.cublasHgemm = 
        (cublasHgemm_t)get_cublas_func("cublasHgemm");
    g_cublas_funcs.cublasGemmEx = 
        (cublasGemmEx_t)get_cublas_func("cublasGemmEx");
    g_cublas_funcs.cublasGemmStridedBatchedEx = 
        (cublasGemmStridedBatchedEx_t)get_cublas_func("cublasGemmStridedBatchedEx");
    g_cublas_funcs.cublasHgemmStridedBatched = 
        (cublasHgemmStridedBatched_t)get_cublas_func("cublasHgemmStridedBatched");
    g_cublas_funcs.cublasSgemmEx = 
        (cublasSgemmEx_t)get_cublas_func("cublasSgemmEx");
    g_cublas_funcs.cublasSetStream_v2 = 
        (cublasSetStream_t)get_cublas_func("cublasSetStream_v2");
    g_cublas_funcs.cublasGetStream_v2 = 
        (cublasGetStream_t)get_cublas_func("cublasGetStream_v2");
    
    g_cublas_funcs.initialized = true;
    if (g_cublas_funcs.cublasSgemm_v2) {
        LOG_DEBUG("cuBLAS functions initialized");
    } else {
        LOG_WARN("cuBLAS functions not found - interception disabled");
    }
}

#define GET_CUBLAS_FUNC(name) \
    do { \
        if (!g_cublas_funcs.initialized) init_cublas_functions(); \
        if (!g_cublas_funcs.name) { \
            LOG_ERROR("Failed to get real " #name); \
            return CUBLAS_STATUS_INTERNAL_ERROR; \
        } \
    } while(0)

// ============================================================================
// cuBLASLt 函数获取和初始化
// ============================================================================

static void* get_cublaslt_func(const char* name) {
    if (!g_cublaslt_handle) {
        const char* lib_paths[] = {
            "libcublasLt.so.12",
            "libcublasLt.so.11", 
            "libcublasLt.so",
            nullptr
        };
        
        for (int i = 0; lib_paths[i]; i++) {
            g_cublaslt_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cublaslt_handle) {
                LOG_DEBUG("Found cuBLASLt library: %s", lib_paths[i]);
                break;
            }
        }
        
        if (!g_cublaslt_handle) {
            for (int i = 0; lib_paths[i]; i++) {
                g_cublaslt_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_cublaslt_handle) {
                    LOG_DEBUG("Loaded cuBLASLt library: %s", lib_paths[i]);
                    break;
                }
            }
        }
    }
    
    if (g_cublaslt_handle) {
        void* fn = dlsym(g_cublaslt_handle, name);
        if (fn) return fn;
    }
    
    return dlsym(RTLD_DEFAULT, name);
}

static void init_cublaslt_functions() {
    std::lock_guard<std::mutex> lock(g_cublaslt_funcs.init_mutex);
    if (g_cublaslt_funcs.initialized) return;
    
    g_cublaslt_funcs.cublasLtMatmul = 
        (cublasLtMatmul_t)get_cublaslt_func("cublasLtMatmul");
    
    g_cublaslt_funcs.initialized = true;
    if (g_cublaslt_funcs.cublasLtMatmul) {
        LOG_DEBUG("cuBLASLt functions initialized");
    } else {
        LOG_WARN("cuBLASLt functions not found - interception disabled");
    }
}

// ============================================================================
// 执行真实 cuBLAS 操作
// ============================================================================

cublasStatus_t execute_cublas_sgemm(OperationPtr op) {
    GET_CUBLAS_FUNC(cublasSgemm_v2);
    auto& p = std::get<CublasGemmParams>(op->params);
    return g_cublas_funcs.cublasSgemm_v2(
        (cublasHandle_t)p.handle,
        (cublasOperation_t)p.transa, (cublasOperation_t)p.transb,
        p.m, p.n, p.k,
        (const float*)p.alpha, (const float*)p.A, p.lda,
        (const float*)p.B, p.ldb,
        (const float*)p.beta, (float*)p.C, p.ldc
    );
}

// 保护 cuBLAS stream 切换的互斥锁
// 注意：只在切换 stream 时加锁，操作本身异步执行以支持并发
static std::mutex g_cublas_stream_mutex;

// 每个调度器线程的独立 cuBLAS handle（用于并发测试）
static thread_local cublasHandle_t tl_cublas_handle = nullptr;
static thread_local bool tl_cublas_handle_initialized = false;

// cuBLAS 创建和设置 stream 的函数指针
using cublasCreate_t = cublasStatus_t (*)(cublasHandle_t*);
using cublasSetStreamDirect_t = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
static cublasCreate_t g_cublasCreate = nullptr;
static cublasSetStreamDirect_t g_cublasSetStreamDirect = nullptr;

// 获取或创建当前线程的 cuBLAS handle
static cublasHandle_t get_thread_local_cublas_handle(cudaStream_t stream) {
    // 延迟初始化函数指针
    if (!g_cublasCreate) {
        g_cublasCreate = (cublasCreate_t)get_cublas_func("cublasCreate_v2");
        g_cublasSetStreamDirect = (cublasSetStreamDirect_t)get_cublas_func("cublasSetStream_v2");
    }
    
    if (!tl_cublas_handle_initialized && g_cublasCreate) {
        cublasStatus_t status = g_cublasCreate(&tl_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            LOG_ERROR("Failed to create thread-local cuBLAS handle: %d", status);
            return nullptr;
        }
        tl_cublas_handle_initialized = true;
        LOG_DEBUG("Created thread-local cuBLAS handle: %p", tl_cublas_handle);
    }
    
    // 设置 stream
    if (tl_cublas_handle && stream && g_cublasSetStreamDirect) {
        g_cublasSetStreamDirect(tl_cublas_handle, stream);
    }
    
    return tl_cublas_handle;
}

// 导出函数：预初始化当前线程的 cuBLAS handle（在调度器线程启动时调用）
void preinit_thread_local_cublas_handle() {
    get_thread_local_cublas_handle(nullptr);
}

cublasStatus_t execute_cublas_operation(OperationPtr op, cudaStream_t scheduler_stream, cublasHandle_t provided_handle) {
    // 确保 cuBLAS 函数已初始化
    if (!g_cublas_funcs.initialized) {
        init_cublas_functions();
    }

    // 设置重入标志，防止 cuBLAS 内部的 cudaLaunchKernel 被再次拦截
    tl_in_scheduler_execution = true;

    cublasStatus_t result = CUBLAS_STATUS_SUCCESS;

    // 优先使用 Green Context 提供的 handle
    cublasHandle_t handle = provided_handle;

    if (!handle) {
        // Fallback: 使用线程本地的 cuBLAS handle
        handle = get_thread_local_cublas_handle(scheduler_stream);
    }

    if (!handle) {
        // 最后回退到原始 handle
        switch (op->type) {
            case OperationType::CUBLAS_SGEMM:
            case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
                auto& p = std::get<CublasGemmParams>(op->params);
                handle = (cublasHandle_t)p.handle;
                // 设置 stream（需要加锁保护）
                if (handle && scheduler_stream && g_cublas_funcs.cublasSetStream_v2) {
                    std::lock_guard<std::mutex> lock(g_cublas_stream_mutex);
                    g_cublas_funcs.cublasSetStream_v2(handle, scheduler_stream);
                }
                break;
            }
            default:
                break;
        }
    } else if (scheduler_stream && g_cublas_funcs.cublasSetStream_v2) {
        // 为提供的 handle 设置 stream
        g_cublas_funcs.cublasSetStream_v2(handle, scheduler_stream);
    }
    
    switch (op->type) {
        case OperationType::CUBLAS_SGEMM: {
            auto& p = std::get<CublasGemmParams>(op->params);
            // 使用存储的标量值或原始指针
            const float* alpha_ptr = p.use_stored_scalars ? &p.alpha_value : (const float*)p.alpha;
            const float* beta_ptr = p.use_stored_scalars ? &p.beta_value : (const float*)p.beta;
            result = g_cublas_funcs.cublasSgemm_v2(
                handle,
                (cublasOperation_t)p.transa,
                (cublasOperation_t)p.transb,
                p.m, p.n, p.k,
                alpha_ptr,
                (const float*)p.A, p.lda,
                (const float*)p.B, p.ldb,
                beta_ptr,
                (float*)p.C, p.ldc
            );
            break;
        }
        
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
            auto& p = std::get<CublasGemmParams>(op->params);
            // 使用存储的标量值或原始指针
            const float* alpha_ptr = p.use_stored_scalars ? &p.alpha_value : (const float*)p.alpha;
            const float* beta_ptr = p.use_stored_scalars ? &p.beta_value : (const float*)p.beta;
            result = g_cublas_funcs.cublasSgemmStridedBatched(
                handle,
                (cublasOperation_t)p.transa,
                (cublasOperation_t)p.transb,
                p.m, p.n, p.k,
                alpha_ptr,
                (const float*)p.A, p.lda, p.strideA,
                (const float*)p.B, p.ldb, p.strideB,
                beta_ptr,
                (float*)p.C, p.ldc, p.strideC,
                p.batchCount
            );
            break;
        }
        
        default:
            LOG_ERROR("Unknown cuBLAS operation type: %d", (int)op->type);
            result = 1;
            break;
    }
    
    // 不同步！让操作异步执行以支持多客户端并发
    // 注意：每个客户端使用独立的 scheduler_stream，操作会在各自的 stream 上执行
    // 客户端通过 wait_operation() 等待操作完成（使用 CUDA event）
    
    tl_in_scheduler_execution = false;
    
    return result;
}

// 直接调用真实函数
cublasStatus_t real_cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc) {
    
    GET_CUBLAS_FUNC(cublasSgemm_v2);
    return g_cublas_funcs.cublasSgemm_v2(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
}

cublasStatus_t real_cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const A[], int lda,
    const float* const B[], int ldb,
    const float* beta,
    float* const C[], int ldc,
    int batchCount) {
    
    GET_CUBLAS_FUNC(cublasSgemmBatched);
    return g_cublas_funcs.cublasSgemmBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc, batchCount
    );
}

cublasStatus_t real_cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    const float* beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    
    GET_CUBLAS_FUNC(cublasSgemmStridedBatched);
    return g_cublas_funcs.cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
    );
}

} // namespace orion

// ============================================================================
// cuBLAS API Wrappers
// ============================================================================

extern "C" {

/**
 * cublasSgemm_v2 wrapper - 单精度 GEMM
 */
// 线程局部重入保护
static thread_local bool tl_in_sgemm = false;

cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc) {
    
    using namespace orion;
    
    // 重入保护：防止递归调用
    if (tl_in_sgemm) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemm_v2) {
            return g_cublas_funcs.cublasSgemm_v2(handle, transa, transb, m, n, k,
                                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    // 如果找不到真实函数，报错
    if (!g_cublas_funcs.cublasSgemm_v2) {
        LOG_ERROR("cublasSgemm_v2 not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemm_v2(handle, transa, transb, m, n, k,
                                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    if (!is_capture_enabled()) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k,
                                    alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k,
                                    alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemm_v2 intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 创建操作并提交到调度器队列
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM);
    if (!op) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k,
                                    alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // 检查异步模式
    bool async_mode = (get_async_mode_internal() == 1);
    
    // 设置 cuBLAS GEMM 参数
    CublasGemmParams params;
    params.handle = handle;
    params.transa = transa;
    params.transb = transb;
    params.m = m;
    params.n = n;
    params.k = k;
    params.A = A;
    params.lda = lda;
    params.B = B;
    params.ldb = ldb;
    params.C = C;
    params.ldc = ldc;
    params.is_batched = false;
    params.is_strided = false;
    
    if (async_mode) {
        // 异步模式：存储 alpha 和 beta 的值
        params.alpha_value = *alpha;
        params.beta_value = *beta;
        params.alpha = &params.alpha_value;
        params.beta = &params.beta_value;
        params.use_stored_scalars = true;
    } else {
        // 同步模式：直接使用指针
        params.alpha = alpha;
        params.beta = beta;
        params.use_stored_scalars = false;
    }
    
    op->params = params;
    
    // 提交到队列
    enqueue_operation(op);
    
    if (async_mode) {
        // 异步模式：不等待，立即返回
        return CUBLAS_STATUS_SUCCESS;
    }
    
    // 同步模式：等待完成
    wait_operation(op);
    return op->result == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1;
}

/**
 * cublasSgemmBatched wrapper - 批量 GEMM
 */
// 线程局部重入保护
static thread_local bool tl_in_batched = false;

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const A[], int lda,
    const float* const B[], int ldb,
    const float* beta,
    float* const C[], int ldc,
    int batchCount) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_batched) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemmBatched) {
            return g_cublas_funcs.cublasSgemmBatched(handle, transa, transb, m, n, k,
                                                      alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasSgemmBatched) {
        LOG_ERROR("cublasSgemmBatched not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemmBatched(handle, transa, transb, m, n, k,
                                                  alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    }
    
    if (!is_capture_enabled()) {
        return real_cublasSgemmBatched(handle, transa, transb, m, n, k,
                                        alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return real_cublasSgemmBatched(handle, transa, transb, m, n, k,
                                        alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemmBatched intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 设置重入标记，防止内部 cudaLaunchKernel 被拦截
    tl_in_batched = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasSgemmBatched(handle, transa, transb, m, n, k,
                                                               alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    tl_in_scheduler_execution = false;
    tl_in_batched = false;
    
    return result;
}

/**
 * cublasSgemmStridedBatched wrapper
 */
// 线程局部重入保护
static thread_local bool tl_in_strided_batched = false;

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    const float* beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_strided_batched) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemmStridedBatched) {
            return g_cublas_funcs.cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                                            alpha, A, lda, strideA, B, ldb, strideB,
                                                            beta, C, ldc, strideC, batchCount);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasSgemmStridedBatched) {
        LOG_ERROR("cublasSgemmStridedBatched not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                                         alpha, A, lda, strideA, B, ldb, strideB,
                                                         beta, C, ldc, strideC, batchCount);
    }
    
    if (!is_capture_enabled()) {
        return real_cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                               alpha, A, lda, strideA, B, ldb, strideB,
                                               beta, C, ldc, strideC, batchCount);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return real_cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                               alpha, A, lda, strideA, B, ldb, strideB,
                                               beta, C, ldc, strideC, batchCount);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemmStridedBatched intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 创建操作并提交到调度器队列
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM_STRIDED_BATCHED);
    if (!op) {
        return real_cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                               alpha, A, lda, strideA, B, ldb, strideB,
                                               beta, C, ldc, strideC, batchCount);
    }
    
    // 检查异步模式
    bool async_mode = (get_async_mode_internal() == 1);
    
    // 设置参数
    CublasGemmParams params;
    params.handle = handle;
    params.transa = transa;
    params.transb = transb;
    params.m = m;
    params.n = n;
    params.k = k;
    params.A = A;
    params.lda = lda;
    params.strideA = strideA;
    params.B = B;
    params.ldb = ldb;
    params.strideB = strideB;
    params.C = C;
    params.ldc = ldc;
    params.strideC = strideC;
    params.batchCount = batchCount;
    params.is_batched = false;
    params.is_strided = true;
    
    if (async_mode) {
        // 异步模式：存储 alpha 和 beta 的值
        params.alpha_value = *alpha;
        params.beta_value = *beta;
        params.use_stored_scalars = true;
    } else {
        // 同步模式：直接使用指针
        params.alpha = alpha;
        params.beta = beta;
        params.use_stored_scalars = false;
    }
    
    op->params = params;
    
    // 提交到队列
    enqueue_operation(op);
    
    if (async_mode) {
        // 异步模式：不等待，立即返回
        return CUBLAS_STATUS_SUCCESS;
    }
    
    // 同步模式：等待完成
    wait_operation(op);
    return op->result == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1;
}

/**
 * cublasGemmEx wrapper - 混合精度 GEMM
 */
static thread_local bool tl_in_gemmex = false;

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta,
    void* C, int Ctype, int ldc,
    int computeType, int algo) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_gemmex) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasGemmEx) {
            return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                                alpha, A, Atype, lda, B, Btype, ldb,
                                                beta, C, Ctype, ldc, computeType, algo);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasGemmEx) {
        LOG_ERROR("cublasGemmEx not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                            alpha, A, Atype, lda, B, Btype, ldb,
                                            beta, C, Ctype, ldc, computeType, algo);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                            alpha, A, Atype, lda, B, Btype, ldb,
                                            beta, C, Ctype, ldc, computeType, algo);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                            alpha, A, Atype, lda, B, Btype, ldb,
                                            beta, C, Ctype, ldc, computeType, algo);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasGemmEx intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_gemmex = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                                         alpha, A, Atype, lda, B, Btype, ldb,
                                                         beta, C, Ctype, ldc, computeType, algo);
    tl_in_scheduler_execution = false;
    tl_in_gemmex = false;
    
    return result;
}

/**
 * cublasGemmStridedBatchedEx wrapper - 混合精度批量 GEMM
 */
static thread_local bool tl_in_gemm_strided_batched_ex = false;

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int Atype, int lda, long long strideA,
    const void* B, int Btype, int ldb, long long strideB,
    const void* beta,
    void* C, int Ctype, int ldc, long long strideC,
    int batchCount,
    int computeType, int algo) {
    
    using namespace orion;
    
    if (tl_in_gemm_strided_batched_ex) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasGemmStridedBatchedEx) {
            return g_cublas_funcs.cublasGemmStridedBatchedEx(
                handle, transa, transb, m, n, k, alpha,
                A, Atype, lda, strideA, B, Btype, ldb, strideB,
                beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasGemmStridedBatchedEx) {
        LOG_ERROR("cublasGemmStridedBatchedEx not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, alpha,
            A, Atype, lda, strideA, B, Btype, ldb, strideB,
            beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, alpha,
            A, Atype, lda, strideA, B, Btype, ldb, strideB,
            beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return g_cublas_funcs.cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, alpha,
            A, Atype, lda, strideA, B, Btype, ldb, strideB,
            beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasGemmStridedBatchedEx intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_gemm_strided_batched_ex = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB,
        beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    tl_in_scheduler_execution = false;
    tl_in_gemm_strided_batched_ex = false;
    
    return result;
}

/**
 * cublasHgemm wrapper - FP16 GEMM
 */
static thread_local bool tl_in_hgemm = false;

cublasStatus_t cublasHgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int lda,
    const void* B, int ldb,
    const void* beta,
    void* C, int ldc) {
    
    using namespace orion;
    
    if (tl_in_hgemm) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasHgemm) {
            return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                               alpha, A, lda, B, ldb, beta, C, ldc);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasHgemm) {
        LOG_ERROR("cublasHgemm not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasHgemm intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_hgemm = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                                        alpha, A, lda, B, ldb, beta, C, ldc);
    tl_in_scheduler_execution = false;
    tl_in_hgemm = false;
    
    return result;
}

/**
 * cublasHgemmStridedBatched wrapper - FP16 批量 GEMM
 */
static thread_local bool tl_in_hgemm_strided_batched = false;

cublasStatus_t cublasHgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int lda, long long strideA,
    const void* B, int ldb, long long strideB,
    const void* beta,
    void* C, int ldc, long long strideC,
    int batchCount) {
    
    using namespace orion;
    
    if (tl_in_hgemm_strided_batched) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasHgemmStridedBatched) {
            return g_cublas_funcs.cublasHgemmStridedBatched(
                handle, transa, transb, m, n, k, alpha,
                A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasHgemmStridedBatched) {
        LOG_ERROR("cublasHgemmStridedBatched not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasHgemmStridedBatched(
            handle, transa, transb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasHgemmStridedBatched(
            handle, transa, transb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return g_cublas_funcs.cublasHgemmStridedBatched(
            handle, transa, transb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasHgemmStridedBatched intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_hgemm_strided_batched = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha,
        A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    tl_in_scheduler_execution = false;
    tl_in_hgemm_strided_batched = false;
    
    return result;
}

/**
 * cublasSgemmEx wrapper - 扩展 FP32 GEMM
 */
static thread_local bool tl_in_sgemmex = false;

cublasStatus_t cublasSgemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const float* beta,
    void* C, int Ctype, int ldc) {
    
    using namespace orion;
    
    if (tl_in_sgemmex) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemmEx) {
            return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                                 alpha, A, Atype, lda, B, Btype, ldb,
                                                 beta, C, Ctype, ldc);
        }
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasSgemmEx) {
        LOG_ERROR("cublasSgemmEx not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                             alpha, A, Atype, lda, B, Btype, ldb,
                                             beta, C, Ctype, ldc);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                             alpha, A, Atype, lda, B, Btype, ldb,
                                             beta, C, Ctype, ldc);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                             alpha, A, Atype, lda, B, Btype, ldb,
                                             beta, C, Ctype, ldc);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemmEx intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_sgemmex = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                                          alpha, A, Atype, lda, B, Btype, ldb,
                                                          beta, C, Ctype, ldc);
    tl_in_scheduler_execution = false;
    tl_in_sgemmex = false;
    
    return result;
}

/**
 * cublasSetStream_v2 wrapper - 拦截 stream 设置
 * 
 * 问题背景：PyTorch 在调用 cuBLAS 计算函数前会调用此函数设置 stream。
 * 如果不拦截，PyTorch 设置的 stream 会覆盖调度器的 stream 控制。
 * 
 * 解决方案：当调度器启用时，忽略 PyTorch 的 stream 设置，
 * 让调度器在执行时设置自己的 stream。
 */
static thread_local bool tl_in_setstream = false;

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    using namespace orion;
    
    // 重入保护
    if (tl_in_setstream || tl_in_scheduler_execution) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSetStream_v2) {
            return g_cublas_funcs.cublasSetStream_v2(handle, streamId);
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) {
        init_cublas_functions();
    }
    
    if (!g_cublas_funcs.cublasSetStream_v2) {
        LOG_ERROR("cublasSetStream_v2 not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 调度器未初始化时，正常透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSetStream_v2(handle, streamId);
    }
    
    // 拦截未启用时，正常透传
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasSetStream_v2(handle, streamId);
    }
    
    // 检查是否有活跃的 client
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        LOG_DEBUG("cublasSetStream_v2 passthrough (no client): stream %p", streamId);
        return g_cublas_funcs.cublasSetStream_v2(handle, streamId);
    }

    // Step B：PyTorch 第一次把 handle 绑到 stream 时，登记 handle→client 以及
    // user-stream→client 的映射。这样即便后续 cuBLAS 内部 worker 线程没有
    // tl_client_idx，也能通过 handle / stream 回查到 client 归属。
    orion::register_client_handle(client_idx, (void*)handle);
    if (streamId) orion::register_client_stream(client_idx, (void*)streamId);

    // 调度器启用时，忽略 PyTorch 的 stream 设置
    // 关键：主动把 cuBLAS handle 的 stream 设为 null，清除 warmup 阶段设置的 stream
    // 这样调度器执行时才能正确设置自己的 stream
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_DEBUG("Client %d: cublasSetStream_v2 intercepted, clearing stream (was %p) (total: %lu)", 
                  client_idx, streamId, count + 1);
    }
    
    // 设置为 null stream，清除之前的 stream 绑定
    tl_in_setstream = true;
    cublasStatus_t result = g_cublas_funcs.cublasSetStream_v2(handle, nullptr);
    tl_in_setstream = false;
    
    return result;
}

/**
 * cublasLtMatmul wrapper - 拦截 cuBLASLt 矩阵乘法
 * 
 * PyTorch 优先使用 cuBLASLt 而不是传统 cuBLAS，stream 直接作为参数传入。
 * 拦截后，调度器执行时使用自己的 stream。
 */
static thread_local bool tl_in_cublaslt_matmul = false;

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void* alpha,
    const void* A,
    cublasLtMatrixLayout_t Adesc,
    const void* B,
    cublasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C,
    cublasLtMatrixLayout_t Cdesc,
    void* D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t* algo,
    void* workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_cublaslt_matmul || tl_in_scheduler_execution) {
        if (!g_cublaslt_funcs.initialized) init_cublaslt_functions();
        if (g_cublaslt_funcs.cublasLtMatmul) {
            return g_cublaslt_funcs.cublasLtMatmul(
                lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
                beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
        }
        return CUBLAS_STATUS_SUCCESS;
    }
    
    // 确保初始化
    if (!g_cublaslt_funcs.initialized) {
        init_cublaslt_functions();
    }
    
    if (!g_cublaslt_funcs.cublasLtMatmul) {
        LOG_ERROR("cublasLtMatmul not found");
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    
    // 调度器未初始化时，正常透传
    if (!g_capture_state.initialized.load()) {
        return g_cublaslt_funcs.cublasLtMatmul(
            lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
    }
    
    // 拦截未启用时，正常透传
    if (!is_capture_enabled()) {
        return g_cublaslt_funcs.cublasLtMatmul(
            lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
    }
    
    // 检查是否有活跃的 client
    int client_idx = resolve_client_idx((void*)stream, (void*)lightHandle);
    if (client_idx < 0) {
        return g_cublaslt_funcs.cublasLtMatmul(
            lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasLtMatmul intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 创建操作并提交到调度器队列
    auto op = create_operation(client_idx, OperationType::CUBLASLT_MATMUL);
    if (!op) {
        return g_cublaslt_funcs.cublasLtMatmul(
            lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
            beta, C, Cdesc, D, Ddesc, algo, workspace, workspaceSizeInBytes, stream);
    }
    
    // 检查异步模式
    bool async_mode = (get_async_mode_internal() == 1);
    
    // 设置 cuBLASLt Matmul 参数
    CublasLtMatmulParams params;
    params.lightHandle = lightHandle;
    params.computeDesc = computeDesc;
    params.A = A;
    params.Adesc = Adesc;
    params.B = B;
    params.Bdesc = Bdesc;
    params.C = C;
    params.Cdesc = Cdesc;
    params.D = D;
    params.Ddesc = Ddesc;
    params.algo = algo;
    params.workspace = workspace;
    params.workspaceSizeInBytes = workspaceSizeInBytes;
    params.stream = stream;  // 保存原始 stream（调度器执行时会替换）
    
    if (async_mode) {
        // 异步模式：存储 alpha 和 beta 的值（假设为 float）
        params.alpha_value = *(const float*)alpha;
        params.beta_value = *(const float*)beta;
        params.use_stored_scalars = true;
    } else {
        // 同步模式：直接使用指针
        params.alpha = alpha;
        params.beta = beta;
        params.use_stored_scalars = false;
    }
    
    op->params = params;

    // 提交到队列
    enqueue_operation(op);

    // 注意：cublasLtMatmul 的 computeDesc / Adesc / Bdesc / Cdesc / Ddesc / algo
    // 都是 PyTorch 在调用前临时创建（cublasLtMatmulDescCreate /
    // cublasLtMatrixLayoutCreate）、调用返回后立即销毁的栈/堆对象。
    // 异步模式下若 client 立即返回，worker 后到时这些描述符已被释放，
    // cuBLASLt 内部 kernel launch 会失败并产生 cudaErrorUnknown(999)。
    // 因此即使在 async_mode==1 下也必须等待 worker 完成。
    // （与 cudaLaunchKernel 因 args 在栈上而强制同步是同一类问题。）
    wait_operation(op);
    return (cublasStatus_t)op->result;
}

} // extern "C"

// ============================================================================
// 执行真实 cuBLASLt 操作（被 scheduler 调用）
// ============================================================================

namespace orion {

cublasStatus_t execute_cublaslt_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    // 确保 cuBLASLt 函数已初始化
    if (!g_cublaslt_funcs.initialized) {
        init_cublaslt_functions();
    }
    
    if (!g_cublaslt_funcs.cublasLtMatmul) {
        LOG_ERROR("cublasLtMatmul not found in execute_cublaslt_operation");
        return (cublasStatus_t)1;
    }
    
    // 设置重入标志
    tl_in_scheduler_execution = true;
    tl_in_cublaslt_matmul = true;
    
    auto& p = std::get<CublasLtMatmulParams>(op->params);
    
    // 使用 scheduler_stream 替换原始 stream
    cudaStream_t exec_stream = scheduler_stream ? scheduler_stream : p.stream;
    
    LOG_DEBUG("Executing cublasLtMatmul on stream %p (original: %p)", exec_stream, p.stream);
    
    // 使用存储的标量值或原始指针
    const void* alpha_ptr = p.use_stored_scalars ? (const void*)&p.alpha_value : p.alpha;
    const void* beta_ptr = p.use_stored_scalars ? (const void*)&p.beta_value : p.beta;
    
    // 不加锁！cuBLASLt 的 handle 和参数都在操作中传递，可以并发执行
    cublasStatus_t result = g_cublaslt_funcs.cublasLtMatmul(
        (cublasLtHandle_t)p.lightHandle,
        (cublasLtMatmulDesc_t)p.computeDesc,
        alpha_ptr,
        p.A, (cublasLtMatrixLayout_t)p.Adesc,
        p.B, (cublasLtMatrixLayout_t)p.Bdesc,
        beta_ptr,
        p.C, (cublasLtMatrixLayout_t)p.Cdesc,
        p.D, (cublasLtMatrixLayout_t)p.Ddesc,
        (const cublasLtMatmulAlgo_t*)p.algo,
        p.workspace,
        p.workspaceSizeInBytes,
        exec_stream);  // 使用调度器的 stream
    
    // 不同步！让操作异步执行以支持多客户端并发
    
    tl_in_cublaslt_matmul = false;
    tl_in_scheduler_execution = false;
    
    return result;
}

} // namespace orion
