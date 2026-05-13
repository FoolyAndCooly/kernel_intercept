/**
 * @file cudnn_intercept.cpp
 * @brief cuDNN API 拦截层实现
 *
 * 本文件实现了 Orion 调度系统的 cuDNN API 拦截功能。
 * cuDNN 是 NVIDIA 的深度神经网络库，提供高度优化的卷积、池化、
 * 归一化等深度学习基础操作。PyTorch、TensorFlow 等框架底层
 * 大量使用 cuDNN 进行神经网络计算。
 *
 * 拦截的 cuDNN API：
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  cuDNN 函数                              │  用途                        │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  cudnnConvolutionForward                 │  卷积前向传播                │
 * │  cudnnConvolutionBackwardData            │  卷积反向传播（数据梯度）    │
 * │  cudnnConvolutionBackwardFilter          │  卷积反向传播（权重梯度）    │
 * │  cudnnBatchNormalizationForwardTraining  │  BatchNorm 训练前向          │
 * │  cudnnBatchNormalizationForwardInference │  BatchNorm 推理前向          │
 * │  cudnnBatchNormalizationBackward         │  BatchNorm 反向传播          │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * 工作流程：
 * 1. 应用程序调用 cuDNN API (如 cudnnConvolutionForward)
 * 2. LD_PRELOAD 将调用重定向到本文件的 wrapper 函数
 * 3. Wrapper 创建 OperationRecord，填充 CudnnConvParams/CudnnBatchNormParams
 * 4. 提交到调度器队列
 * 5. 调度器线程调用 execute_cudnn_operation() 执行
 * 6. 返回执行结果
 *
 * 注意事项：
 * - cuDNN 操作通常是 compute-bound（卷积）或 memory-bound（BatchNorm）
 * - 这些信息用于 Orion 的 profile 互补调度决策
 * - 当前实现未设置 cudnnSetStream，TODO: 添加 stream 控制
 */

#include "gpu_capture.h"
#include <dlfcn.h>
#include <cstdio>
#include <mutex>

// ============================================================================
// cuDNN 类型定义
// ============================================================================
// 避免在此文件中引入 cuDNN 头文件依赖，手动定义所需类型。
// 这些类型与 cuDNN 库中的定义兼容。

typedef void* cudnnHandle_t;                    // cuDNN 句柄类型
typedef void* cudnnTensorDescriptor_t;          // 张量描述符
typedef void* cudnnFilterDescriptor_t;          // 卷积核描述符
typedef void* cudnnConvolutionDescriptor_t;     // 卷积操作描述符
typedef void* cudnnActivationDescriptor_t;      // 激活函数描述符
typedef void* cudnnBatchNormMode_t;             // BatchNorm 模式
typedef int cudnnStatus_t;                      // cuDNN 状态/错误码类型
typedef int cudnnConvolutionFwdAlgo_t;          // 卷积前向算法类型
typedef int cudnnConvolutionBwdDataAlgo_t;      // 卷积反向（数据）算法类型
typedef int cudnnConvolutionBwdFilterAlgo_t;    // 卷积反向（权重）算法类型

#define CUDNN_STATUS_SUCCESS 0                  // 成功状态码

namespace orion {

// ============================================================================
// cuDNN 真实函数指针类型
// ============================================================================
// 定义所有需要拦截的 cuDNN API 的函数指针类型。
// 这些类型用于存储通过 dlsym 获取的真实函数地址。

/**
 * @brief cudnnConvolutionForward 函数指针类型
 *
 * 卷积前向传播：y = alpha * conv(x, w) + beta * y
 * 参数：handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workspace, workspaceSize, beta, yDesc, y
 */
using cudnnConvolutionForward_t = cudnnStatus_t (*)(
    cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnFilterDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    cudnnConvolutionFwdAlgo_t, void*, size_t, const void*,
    cudnnTensorDescriptor_t, void*);

/**
 * @brief cudnnConvolutionBackwardData 函数指针类型
 *
 * 卷积反向传播（计算输入数据的梯度）：dx = alpha * conv_bwd(w, dy) + beta * dx
 * 用于神经网络训练时的反向传播
 */
using cudnnConvolutionBackwardData_t = cudnnStatus_t (*)(
    cudnnHandle_t, const void*, cudnnFilterDescriptor_t, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    cudnnConvolutionBwdDataAlgo_t, void*, size_t, const void*,
    cudnnTensorDescriptor_t, void*);

/**
 * @brief cudnnConvolutionBackwardFilter 函数指针类型
 *
 * 卷积反向传播（计算卷积核权重的梯度）：dw = alpha * conv_bwd(x, dy) + beta * dw
 * 用于神经网络训练时更新卷积核权重
 */
using cudnnConvolutionBackwardFilter_t = cudnnStatus_t (*)(
    cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    cudnnConvolutionBwdFilterAlgo_t, void*, size_t, const void*,
    cudnnFilterDescriptor_t, void*);

/**
 * @brief cudnnBatchNormalizationForwardTraining 函数指针类型
 *
 * BatchNorm 训练前向传播：
 * - 计算当前 batch 的均值和方差
 * - 更新 running mean 和 running variance
 * - 输出归一化后的结果
 */
using cudnnBatchNormalizationForwardTraining_t = cudnnStatus_t (*)(
    cudnnHandle_t, int, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
    cudnnTensorDescriptor_t, const void*, const void*,
    double, void*, void*, double, void*, void*);

/**
 * @brief cudnnBatchNormalizationForwardInference 函数指针类型
 *
 * BatchNorm 推理前向传播：
 * - 使用预先计算的 running mean 和 running variance
 * - 不更新统计量
 */
using cudnnBatchNormalizationForwardInference_t = cudnnStatus_t (*)(
    cudnnHandle_t, int, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
    cudnnTensorDescriptor_t, const void*, const void*,
    const void*, const void*, double);

/**
 * @brief cudnnBatchNormalizationBackward 函数指针类型
 *
 * BatchNorm 反向传播：
 * - 计算输入数据的梯度 dx
 * - 计算 scale 参数的梯度 dBnScaleResult
 * - 计算 bias 参数的梯度 dBnBiasResult
 */
using cudnnBatchNormalizationBackward_t = cudnnStatus_t (*)(
    cudnnHandle_t, int, const void*, const void*, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*,
    void*, void*, double, const void*, const void*);

// Step B：拦截 cudnnSetStream，维护 handle↔stream↔client 映射
using cudnnSetStream_t  = cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t);
using cudnnGetStream_t  = cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t*);

// ============================================================================
// 真实函数指针存储
// ============================================================================
// 存储通过 dlsym 获取的真实 cuDNN 函数指针。
// 使用延迟初始化，在第一次调用时获取函数地址。

/**
 * @brief 真实 cuDNN 函数指针结构体
 *
 * 包含所有被拦截的 cuDNN API 的真实函数指针。
 * initialized 标志确保只初始化一次。
 */
static struct {
    cudnnConvolutionForward_t cudnnConvolutionForward;
    cudnnConvolutionBackwardData_t cudnnConvolutionBackwardData;
    cudnnConvolutionBackwardFilter_t cudnnConvolutionBackwardFilter;
    cudnnBatchNormalizationForwardTraining_t cudnnBatchNormalizationForwardTraining;
    cudnnBatchNormalizationForwardInference_t cudnnBatchNormalizationForwardInference;
    cudnnBatchNormalizationBackward_t cudnnBatchNormalizationBackward;
    cudnnSetStream_t cudnnSetStream;
    cudnnGetStream_t cudnnGetStream;
    bool initialized;           // 是否已初始化
    std::mutex init_mutex;      // 保护初始化过程的互斥锁
} g_cudnn_funcs = {nullptr};

/**
 * @brief 获取 cuDNN 函数指针
 *
 * 通过 dlsym 从 cuDNN 库中获取函数指针。
 *
 * 查找顺序：
 * 1. RTLD_NEXT: 查找下一个同名符号（跳过当前拦截）
 * 2. RTLD_DEFAULT: 从默认符号表查找
 *
 * @param name 函数名称
 * @return 函数指针，失败返回 nullptr
 */
static void* get_cudnn_func(const char* name) {
    // 首先尝试 RTLD_NEXT，获取下一个同名符号
    void* fn = dlsym(RTLD_NEXT, name);
    if (fn) return fn;
    // 备选：从默认符号表查找
    fn = dlsym(RTLD_DEFAULT, name);
    return fn;
}

/**
 * @brief 初始化真实 cuDNN 函数指针
 *
 * 获取所有被拦截的 cuDNN API 的真实函数指针。
 * 使用互斥锁确保线程安全，只初始化一次。
 */
static void init_cudnn_functions() {
    std::lock_guard<std::mutex> lock(g_cudnn_funcs.init_mutex);
    if (g_cudnn_funcs.initialized) return;

    // 获取卷积相关函数
    g_cudnn_funcs.cudnnConvolutionForward =
        (cudnnConvolutionForward_t)get_cudnn_func("cudnnConvolutionForward");
    g_cudnn_funcs.cudnnConvolutionBackwardData =
        (cudnnConvolutionBackwardData_t)get_cudnn_func("cudnnConvolutionBackwardData");
    g_cudnn_funcs.cudnnConvolutionBackwardFilter =
        (cudnnConvolutionBackwardFilter_t)get_cudnn_func("cudnnConvolutionBackwardFilter");

    // 获取 BatchNorm 相关函数
    g_cudnn_funcs.cudnnBatchNormalizationForwardTraining =
        (cudnnBatchNormalizationForwardTraining_t)get_cudnn_func("cudnnBatchNormalizationForwardTraining");
    g_cudnn_funcs.cudnnBatchNormalizationForwardInference =
        (cudnnBatchNormalizationForwardInference_t)get_cudnn_func("cudnnBatchNormalizationForwardInference");
    g_cudnn_funcs.cudnnBatchNormalizationBackward =
        (cudnnBatchNormalizationBackward_t)get_cudnn_func("cudnnBatchNormalizationBackward");

    // Step B：stream 绑定观测
    g_cudnn_funcs.cudnnSetStream =
        (cudnnSetStream_t)get_cudnn_func("cudnnSetStream");
    g_cudnn_funcs.cudnnGetStream =
        (cudnnGetStream_t)get_cudnn_func("cudnnGetStream");

    g_cudnn_funcs.initialized = true;
    LOG_DEBUG("cuDNN functions initialized");
}

/**
 * @brief 获取真实函数宏（带延迟初始化）
 *
 * 在调用真实 cuDNN 函数前使用此宏确保函数指针已初始化。
 * 如果获取失败，返回错误码 1。
 *
 * @param name 函数名称
 */
#define GET_CUDNN_FUNC(name) \
    do { \
        if (!g_cudnn_funcs.initialized) init_cudnn_functions(); \
        if (!g_cudnn_funcs.name) { \
            LOG_ERROR("Failed to get real " #name); \
            return 1; \
        } \
    } while(0)

// ============================================================================
// 执行真实 cuDNN 操作 (由调度器调用)
// ============================================================================
// 这些函数在调度器线程中执行，调用真实的 cuDNN API。
// 它们从 OperationRecord 中提取参数，执行操作，并返回结果。

/**
 * @brief 执行 cudnnConvolutionForward 操作
 *
 * 从 OperationRecord 中提取 CudnnConvParams，调用真实的卷积前向函数。
 *
 * @param op 操作记录（包含 CudnnConvParams）
 * @return cuDNN 状态码
 */
cudnnStatus_t execute_cudnn_conv_fwd(OperationPtr op) {
    GET_CUDNN_FUNC(cudnnConvolutionForward);
    auto& p = std::get<CudnnConvParams>(op->params);
    return g_cudnn_funcs.cudnnConvolutionForward(
        (cudnnHandle_t)p.handle, p.alpha,
        (cudnnTensorDescriptor_t)p.xDesc, p.x,
        (cudnnFilterDescriptor_t)p.wDesc, p.w,
        (cudnnConvolutionDescriptor_t)p.convDesc,
        (cudnnConvolutionFwdAlgo_t)p.algo, p.workSpace, p.workSpaceSizeInBytes,
        p.beta, (cudnnTensorDescriptor_t)p.yDesc, p.y
    );
}

/**
 * @brief 执行 cuDNN 操作的统一入口（导出给调度器使用）
 *
 * 这是调度器调用的主要接口，根据操作类型分发到具体的执行函数。
 *
 * TODO: 使用 cudnnSetStream 设置调度器的 stream，实现 stream 控制
 *
 * @param op 操作记录
 * @param scheduler_stream 调度器分配的 CUDA stream（当前未使用）
 * @return cuDNN 状态码
 */
cudnnStatus_t execute_cudnn_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    // TODO: 使用 cudnnSetStream 设置调度器的 stream
    (void)scheduler_stream;

    switch (op->type) {
        case OperationType::CUDNN_CONV_FWD:
            return execute_cudnn_conv_fwd(op);
        // 其他 cuDNN 操作类似处理
        // TODO: 添加 CUDNN_CONV_BWD_DATA, CUDNN_CONV_BWD_FILTER, CUDNN_BATCHNORM_* 的执行
        default:
            LOG_ERROR("Unknown cuDNN operation type");
            return 1;
    }
}

// ============================================================================
// 直接调用真实函数的版本（用于非管理线程）
// ============================================================================
// 这些函数直接调用真实的 cuDNN API，不经过调度器。
// 用于非管理线程（未注册的线程）或拦截未启用时。

/**
 * @brief 直接调用真实的 cudnnConvolutionForward
 *
 * 卷积前向传播：y = alpha * conv(x, w) + beta * y
 */
cudnnStatus_t real_cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y) {

    GET_CUDNN_FUNC(cudnnConvolutionForward);
    return g_cudnn_funcs.cudnnConvolutionForward(
        handle, alpha, xDesc, x, wDesc, w, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y
    );
}

/**
 * @brief 直接调用真实的 cudnnConvolutionBackwardData
 *
 * 卷积反向传播（数据梯度）：dx = alpha * conv_bwd(w, dy) + beta * dx
 */
cudnnStatus_t real_cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx) {

    GET_CUDNN_FUNC(cudnnConvolutionBackwardData);
    return g_cudnn_funcs.cudnnConvolutionBackwardData(
        handle, alpha, wDesc, w, dyDesc, dy, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx
    );
}

/**
 * @brief 直接调用真实的 cudnnConvolutionBackwardFilter
 *
 * 卷积反向传播（权重梯度）：dw = alpha * conv_bwd(x, dy) + beta * dw
 */
cudnnStatus_t real_cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnFilterDescriptor_t dwDesc, void* dw) {

    GET_CUDNN_FUNC(cudnnConvolutionBackwardFilter);
    return g_cudnn_funcs.cudnnConvolutionBackwardFilter(
        handle, alpha, xDesc, x, dyDesc, dy, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw
    );
}

/**
 * @brief 直接调用真实的 cudnnBatchNormalizationForwardTraining
 *
 * BatchNorm 训练前向传播：
 * - 计算 batch 均值和方差
 * - 更新 running statistics
 * - 输出归一化结果
 */
cudnnStatus_t real_cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, int mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon, void* resultSaveMean, void* resultSaveInvVariance) {

    GET_CUDNN_FUNC(cudnnBatchNormalizationForwardTraining);
    return g_cudnn_funcs.cudnnBatchNormalizationForwardTraining(
        handle, mode, alpha, beta, xDesc, x, yDesc, y,
        bnScaleBiasMeanVarDesc, bnScale, bnBias,
        exponentialAverageFactor, resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance
    );
}

} // namespace orion

// ============================================================================
// cuDNN API Wrappers (LD_PRELOAD 拦截点)
// ============================================================================
// 这些函数是 LD_PRELOAD 的拦截点，与真实的 cuDNN API 同名。
// 当应用程序调用 cuDNN API 时，会被重定向到这些 wrapper 函数。
//
// 每个 wrapper 函数的通用流程：
// 1. 检查拦截是否启用，未启用则直接透传
// 2. 获取当前线程的客户端索引
// 3. 创建 OperationRecord 并填充参数
// 4. 提交到队列并等待完成
// 5. 返回执行结果

extern "C" {

/**
 * @brief cudnnConvolutionForward 拦截 wrapper
 *
 * 拦截卷积前向传播操作。
 * 卷积是深度学习中最常用的操作之一，通常是 compute-bound。
 *
 * @param handle cuDNN 句柄
 * @param alpha 缩放因子（输出 = alpha * conv_result + beta * y）
 * @param xDesc 输入张量描述符
 * @param x 输入数据指针
 * @param wDesc 卷积核描述符
 * @param w 卷积核数据指针
 * @param convDesc 卷积操作描述符
 * @param algo 卷积算法
 * @param workSpace 工作空间指针
 * @param workSpaceSizeInBytes 工作空间大小
 * @param beta 缩放因子
 * @param yDesc 输出张量描述符
 * @param y 输出数据指针
 * @return cuDNN 状态码
 */
cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnConvolutionForward(
            handle, alpha, xDesc, x, wDesc, w, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        LOG_INFO("[CUDNN] cudnnConvolutionForward: client_idx < 0, bypassing");
        return real_cudnnConvolutionForward(
            handle, alpha, xDesc, x, wDesc, w, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    }

    if (client_idx == 1) {
        LOG_INFO("[CUDNN] Client 1: cudnnConvolutionForward intercepted");
    }

    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_FWD);
    if (!op) {
        LOG_INFO("[CUDNN] Client %d: submit_operation returned nullptr!", client_idx);
        return 1;
    }

    if (client_idx == 1) {
        LOG_INFO("[CUDNN] Client 1: op created, op_id=%lu", op->op_id);
    }
    
    CudnnConvParams p;
    p.handle = handle;
    p.alpha = alpha;
    p.xDesc = xDesc;
    p.x = x;
    p.wDesc = wDesc;
    p.w = w;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workSpace;
    p.workSpaceSizeInBytes = workSpaceSizeInBytes;
    p.beta = beta;
    p.yDesc = yDesc;
    p.y = y;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

/**
 * @brief cudnnConvolutionBackwardData 拦截 wrapper
 *
 * 拦截卷积反向传播（数据梯度）操作。
 * 用于计算输入数据的梯度，是神经网络训练反向传播的关键步骤。
 *
 * @param handle cuDNN 句柄
 * @param alpha 缩放因子
 * @param wDesc 卷积核描述符
 * @param w 卷积核数据
 * @param dyDesc 输出梯度描述符
 * @param dy 输出梯度数据
 * @param convDesc 卷积操作描述符
 * @param algo 反向传播算法
 * @param workSpace 工作空间
 * @param workSpaceSizeInBytes 工作空间大小
 * @param beta 缩放因子
 * @param dxDesc 输入梯度描述符
 * @param dx 输入梯度数据（输出）
 * @return cuDNN 状态码
 */
cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnConvolutionBackwardData(
            handle, alpha, wDesc, w, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return real_cudnnConvolutionBackwardData(
            handle, alpha, wDesc, w, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    }
    
    LOG_TRACE("Client %d: cudnnConvolutionBackwardData", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_BWD_DATA);
    if (!op) return 1;
    
    CudnnConvParams p;
    p.handle = handle;
    p.alpha = alpha;
    p.wDesc = wDesc;
    p.w = w;
    p.xDesc = dyDesc;
    p.x = dy;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workSpace;
    p.workSpaceSizeInBytes = workSpaceSizeInBytes;
    p.beta = beta;
    p.yDesc = dxDesc;
    p.y = dx;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

/**
 * @brief cudnnConvolutionBackwardFilter 拦截 wrapper
 *
 * 拦截卷积反向传播（权重梯度）操作。
 * 用于计算卷积核权重的梯度，是神经网络训练更新权重的关键步骤。
 *
 * @param handle cuDNN 句柄
 * @param alpha 缩放因子
 * @param xDesc 输入张量描述符
 * @param x 输入数据
 * @param dyDesc 输出梯度描述符
 * @param dy 输出梯度数据
 * @param convDesc 卷积操作描述符
 * @param algo 反向传播算法
 * @param workSpace 工作空间
 * @param workSpaceSizeInBytes 工作空间大小
 * @param beta 缩放因子
 * @param dwDesc 权重梯度描述符
 * @param dw 权重梯度数据（输出）
 * @return cuDNN 状态码
 */
cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnFilterDescriptor_t dwDesc, void* dw) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnConvolutionBackwardFilter(
            handle, alpha, xDesc, x, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return real_cudnnConvolutionBackwardFilter(
            handle, alpha, xDesc, x, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    }
    
    LOG_TRACE("Client %d: cudnnConvolutionBackwardFilter", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_BWD_FILTER);
    if (!op) return 1;
    
    CudnnConvParams p;
    p.handle = handle;
    p.alpha = alpha;
    p.xDesc = xDesc;
    p.x = x;
    p.yDesc = dyDesc;
    p.y = (void*)dy;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workSpace;
    p.workSpaceSizeInBytes = workSpaceSizeInBytes;
    p.beta = beta;
    p.wDesc = dwDesc;
    p.w = dw;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

/**
 * @brief cudnnBatchNormalizationForwardTraining 拦截 wrapper
 *
 * 拦截 BatchNorm 训练前向传播操作。
 * BatchNorm 是深度学习中常用的归一化层，通常是 memory-bound 操作。
 *
 * 功能：
 * - 计算当前 batch 的均值和方差
 * - 使用指数移动平均更新 running mean 和 running variance
 * - 输出归一化后的结果
 *
 * @param handle cuDNN 句柄
 * @param mode BatchNorm 模式（per-activation 或 spatial）
 * @param alpha 缩放因子
 * @param beta 缩放因子
 * @param xDesc 输入张量描述符
 * @param x 输入数据
 * @param yDesc 输出张量描述符
 * @param y 输出数据
 * @param bnScaleBiasMeanVarDesc scale/bias/mean/var 描述符
 * @param bnScale 归一化 scale 参数
 * @param bnBias 归一化 bias 参数
 * @param exponentialAverageFactor 指数移动平均因子
 * @param resultRunningMean 更新后的 running mean（输出）
 * @param resultRunningVariance 更新后的 running variance（输出）
 * @param epsilon 数值稳定性参数
 * @param resultSaveMean 保存的 batch mean（用于反向传播）
 * @param resultSaveInvVariance 保存的 batch inverse variance（用于反向传播）
 * @return cuDNN 状态码
 */
cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, int mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon, void* resultSaveMean, void* resultSaveInvVariance) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnBatchNormalizationForwardTraining(
            handle, mode, alpha, beta, xDesc, x, yDesc, y,
            bnScaleBiasMeanVarDesc, bnScale, bnBias,
            exponentialAverageFactor, resultRunningMean, resultRunningVariance,
            epsilon, resultSaveMean, resultSaveInvVariance);
    }
    
    int client_idx = resolve_client_idx(nullptr, (void*)handle);
    if (client_idx < 0) {
        return real_cudnnBatchNormalizationForwardTraining(
            handle, mode, alpha, beta, xDesc, x, yDesc, y,
            bnScaleBiasMeanVarDesc, bnScale, bnBias,
            exponentialAverageFactor, resultRunningMean, resultRunningVariance,
            epsilon, resultSaveMean, resultSaveInvVariance);
    }
    
    LOG_TRACE("Client %d: cudnnBatchNormalizationForwardTraining", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_BATCHNORM_FWD);
    if (!op) return 1;
    
    CudnnBatchNormParams p;
    p.handle = handle;
    p.mode = mode;
    p.alpha = alpha;
    p.beta = beta;
    p.xDesc = xDesc;
    p.x = x;
    p.yDesc = yDesc;
    p.y = y;
    p.bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc;
    p.bnScale = bnScale;
    p.bnBias = bnBias;
    p.exponentialAverageFactor = exponentialAverageFactor;
    p.resultRunningMean = resultRunningMean;
    p.resultRunningVariance = resultRunningVariance;
    p.epsilon = epsilon;
    p.resultSaveMean = resultSaveMean;
    p.resultSaveInvVariance = resultSaveInvVariance;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

// ============================================================================
// Step B: cudnnSetStream 拦截
// ============================================================================
// PyTorch / cuDNN 会在每个 handle 上调用 cudnnSetStream 把 handle 绑到用户
// stream。拦截此调用是为了：
//   1) 记录 handle↔client 的映射，让 cuDNN 内部 worker 线程（没有
//      tl_client_idx）在发 launch 时仍能通过 handle 反查到 client 归属；
//   2) 顺带把传入的 stream 也登记到 stream→client 映射，覆盖 capture 启用
//      前就已经创建、但没走 register_client_stream 的 stream。
// 行为：始终透传到真实 cudnnSetStream，不改写其语义。调度器执行 op 时会
// 通过 cudnnSetStream 自己再把 handle 绑到调度器 stream，不影响这里。
// ----------------------------------------------------------------------------
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    using namespace orion;
    if (!g_cudnn_funcs.initialized) init_cudnn_functions();
    if (!g_cudnn_funcs.cudnnSetStream) {
        return (cudnnStatus_t)1;  // CUDNN_STATUS_NOT_INITIALIZED
    }

    // 仅在调度器已初始化且有明确 client 归属时登记映射
    if (g_capture_state.initialized.load()) {
        int client_idx = resolve_client_idx((void*)streamId, (void*)handle);
        if (client_idx >= 0) {
            register_client_handle(client_idx, (void*)handle);
            if (streamId) register_client_stream(client_idx, (void*)streamId);
        }
    }

    return g_cudnn_funcs.cudnnSetStream(handle, streamId);
}

} // extern "C"
