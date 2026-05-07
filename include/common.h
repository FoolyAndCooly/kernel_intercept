/**
 * @file common.h
 * @brief Orion 调度器公共定义文件
 *
 * 本文件定义了 Orion GPU 调度系统的公共类型、枚举和工具函数。
 * 包括：
 * - 操作类型枚举 (OperationType): 定义所有可被拦截的 CUDA/cuBLAS/cuDNN 操作
 * - Profile 类型枚举 (ProfileType): 用于区分 compute-bound 和 memory-bound kernel
 * - 日志系统: 支持多级别日志输出
 * - 线程局部变量声明: 用于区分调度器线程和客户端线程
 */

#ifndef ORION_COMMON_H
#define ORION_COMMON_H

#include <cstdint>
#include <atomic>
#include <string>

namespace orion {

// ============================================================================
// 常量定义
// ============================================================================

/**
 * @brief 最大支持的客户端数量
 *
 * Orion 调度器支持同时管理多个客户端（如 1 个 HP + 多个 BE）。
 * 每个客户端有独立的操作队列和调度器线程。
 */
constexpr int MAX_CLIENTS = 16;

// ============================================================================
// 操作类型枚举
// ============================================================================

/**
 * @brief CUDA 操作类型枚举
 *
 * 定义了所有可被 Orion 拦截和调度的 CUDA 操作类型。
 * 这些操作通过 LD_PRELOAD 机制被拦截，然后提交到调度器队列。
 *
 * 操作分类：
 * 1. Kernel 类操作 (会占用 GPU SM):
 *    - KERNEL_LAUNCH: cudaLaunchKernel
 *    - CUDNN_*: cuDNN 卷积、BatchNorm 等
 *    - CUBLAS_*: cuBLAS 矩阵运算
 *
 * 2. 内存操作 (不占用 SM，通常直接执行):
 *    - MALLOC/FREE: cudaMalloc/cudaFree
 *    - MEMCPY/MEMCPY_ASYNC: cudaMemcpy/cudaMemcpyAsync
 *    - MEMSET/MEMSET_ASYNC: cudaMemset/cudaMemsetAsync
 *
 * 3. 同步操作:
 *    - DEVICE_SYNC: cudaDeviceSynchronize
 *    - STREAM_SYNC: cudaStreamSynchronize
 *    - EVENT_SYNC: cudaEventSynchronize
 */
enum class OperationType : uint8_t {
    // === Kernel 启动 ===
    KERNEL_LAUNCH = 0,      // cudaLaunchKernel - 原生 CUDA kernel 启动

    // === 内存管理 ===
    MALLOC,                 // cudaMalloc - GPU 内存分配
    FREE,                   // cudaFree - GPU 内存释放
    MEMCPY,                 // cudaMemcpy - 同步内存拷贝
    MEMCPY_ASYNC,           // cudaMemcpyAsync - 异步内存拷贝
    MEMSET,                 // cudaMemset - 同步内存设置
    MEMSET_ASYNC,           // cudaMemsetAsync - 异步内存设置

    // === 同步操作 ===
    DEVICE_SYNC,            // cudaDeviceSynchronize - 设备级同步
    STREAM_SYNC,            // cudaStreamSynchronize - Stream 级同步
    EVENT_SYNC,             // cudaEventSynchronize - Event 级同步

    // === cuDNN 操作 (深度学习卷积库) ===
    CUDNN_CONV_FWD,         // cudnnConvolutionForward - 卷积前向
    CUDNN_CONV_BWD_DATA,    // cudnnConvolutionBackwardData - 卷积反向(数据)
    CUDNN_CONV_BWD_FILTER,  // cudnnConvolutionBackwardFilter - 卷积反向(权重)
    CUDNN_BATCHNORM_FWD,    // cudnnBatchNormalizationForwardTraining - BN 前向
    CUDNN_BATCHNORM_BWD,    // cudnnBatchNormalizationBackward - BN 反向

    // === cuBLAS 操作 (线性代数库) ===
    CUBLAS_SGEMM,                   // cublasSgemm - 单精度矩阵乘法
    CUBLAS_SGEMM_BATCHED,           // cublasSgemmBatched - 批量矩阵乘法
    CUBLAS_SGEMM_STRIDED_BATCHED,   // cublasSgemmStridedBatched - 跨步批量矩阵乘法
    CUBLASLT_MATMUL,                // cublasLtMatmul - cuBLASLt 矩阵乘法

    UNKNOWN                 // 未知操作类型
};

// ============================================================================
// Profile 类型枚举
// ============================================================================

/**
 * @brief Kernel Profile 类型
 *
 * 用于 Orion 调度决策中的 "Profile 互补性" 判断。
 *
 * 调度策略：
 * - COMPUTE_BOUND + MEMORY_BOUND 可以并发执行（互补）
 * - COMPUTE_BOUND + COMPUTE_BOUND 会竞争 SM，应避免并发
 * - MEMORY_BOUND + MEMORY_BOUND 会竞争内存带宽，应避免并发
 *
 * Profile 信息从 kernel_info.csv 文件加载，由 NCU profiling 生成。
 */
enum class ProfileType : uint8_t {
    COMPUTE_BOUND = 0,  // 计算密集型: 主要瓶颈在 SM 计算能力
    MEMORY_BOUND,       // 内存密集型: 主要瓶颈在内存带宽
    UNKNOWN             // 未知类型: 无法确定或未 profile
};

// ============================================================================
// 操作类型名称转换
// ============================================================================

/**
 * @brief 获取操作类型的字符串名称
 *
 * 用于日志输出和调试。
 *
 * @param type 操作类型枚举值
 * @return 操作类型的字符串表示
 */
inline const char* op_type_name(OperationType type) {
    switch (type) {
        case OperationType::KERNEL_LAUNCH: return "KERNEL_LAUNCH";
        case OperationType::MALLOC: return "MALLOC";
        case OperationType::FREE: return "FREE";
        case OperationType::MEMCPY: return "MEMCPY";
        case OperationType::MEMCPY_ASYNC: return "MEMCPY_ASYNC";
        case OperationType::MEMSET: return "MEMSET";
        case OperationType::MEMSET_ASYNC: return "MEMSET_ASYNC";
        case OperationType::DEVICE_SYNC: return "DEVICE_SYNC";
        case OperationType::STREAM_SYNC: return "STREAM_SYNC";
        case OperationType::EVENT_SYNC: return "EVENT_SYNC";
        case OperationType::CUDNN_CONV_FWD: return "CUDNN_CONV_FWD";
        case OperationType::CUDNN_CONV_BWD_DATA: return "CUDNN_CONV_BWD_DATA";
        case OperationType::CUDNN_CONV_BWD_FILTER: return "CUDNN_CONV_BWD_FILTER";
        case OperationType::CUDNN_BATCHNORM_FWD: return "CUDNN_BATCHNORM_FWD";
        case OperationType::CUDNN_BATCHNORM_BWD: return "CUDNN_BATCHNORM_BWD";
        case OperationType::CUBLAS_SGEMM: return "CUBLAS_SGEMM";
        case OperationType::CUBLAS_SGEMM_BATCHED: return "CUBLAS_SGEMM_BATCHED";
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: return "CUBLAS_SGEMM_STRIDED_BATCHED";
        case OperationType::CUBLASLT_MATMUL: return "CUBLASLT_MATMUL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// 日志系统
// ============================================================================

/**
 * @brief 日志级别枚举
 *
 * 从低到高：NONE < ERROR < WARN < INFO < DEBUG < TRACE
 * 设置某个级别后，该级别及以下的日志都会输出。
 *
 * 可通过环境变量 ORION_LOG_LEVEL 设置：
 * - ORION_LOG_LEVEL=NONE  或 0: 不输出任何日志
 * - ORION_LOG_LEVEL=ERROR 或 1: 只输出错误
 * - ORION_LOG_LEVEL=WARN  或 2: 输出警告和错误
 * - ORION_LOG_LEVEL=INFO  或 3: 输出信息、警告和错误 (默认)
 * - ORION_LOG_LEVEL=DEBUG 或 4: 输出调试信息
 * - ORION_LOG_LEVEL=TRACE 或 5: 输出所有详细跟踪信息
 */
enum class LogLevel : uint8_t {
    NONE = 0,   // 不输出任何日志
    ERROR,      // 错误: 严重问题，可能导致功能失败
    WARN,       // 警告: 潜在问题，但不影响主要功能
    INFO,       // 信息: 重要的运行状态信息
    DEBUG,      // 调试: 详细的调试信息
    TRACE       // 跟踪: 最详细的跟踪信息，包括每个操作
};

/**
 * @brief 全局日志级别
 *
 * 在 gpu_capture.cpp 中定义，通过 init_log_level() 初始化。
 * 默认值为 INFO。
 */
extern LogLevel g_log_level;

/**
 * @brief 初始化日志级别
 *
 * 从环境变量 ORION_LOG_LEVEL 读取日志级别设置。
 * 在 gpu_capture.cpp 中实现。
 */
void init_log_level();

/**
 * @brief 日志输出宏
 *
 * 根据当前日志级别决定是否输出日志。
 * 输出格式: [ORION][LEVEL] message
 *
 * @param level 日志级别
 * @param fmt printf 格式字符串
 * @param ... 格式化参数
 */
#define ORION_LOG(level, fmt, ...) \
    do { \
        if (static_cast<uint8_t>(level) <= static_cast<uint8_t>(orion::g_log_level)) { \
            fprintf(stderr, "[ORION][%s] " fmt "\n", \
                    #level, ##__VA_ARGS__); \
            fflush(stderr); \
        } \
    } while(0)

// 便捷日志宏
#define LOG_ERROR(fmt, ...) ORION_LOG(orion::LogLevel::ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  ORION_LOG(orion::LogLevel::WARN, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  ORION_LOG(orion::LogLevel::INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) ORION_LOG(orion::LogLevel::DEBUG, fmt, ##__VA_ARGS__)
#define LOG_TRACE(fmt, ...) ORION_LOG(orion::LogLevel::TRACE, fmt, ##__VA_ARGS__)

// ============================================================================
// 线程局部变量声明
// ============================================================================
// 这些变量在 gpu_capture.cpp 中定义，用于区分不同类型的线程

/**
 * @brief 标识当前线程是否为调度器线程
 *
 * 工作原理：
 * - 客户端线程: tl_is_scheduler_thread = false
 *   → 所有 CUDA 调用被拦截，提交到调度器队列
 * - 调度器线程: tl_is_scheduler_thread = true
 *   → CUDA 调用直接执行，不被拦截（避免无限递归）
 *
 * 设置时机：
 * - 调度器线程在 run_client() 开始时设置为 true
 * - 客户端线程保持默认值 false
 */
extern thread_local bool tl_is_scheduler_thread;

/**
 * @brief 重入保护标志
 *
 * 当调度器线程正在执行某个操作时设置为 true。
 * 用于防止执行过程中产生的递归拦截。
 *
 * 例如：执行 cuBLAS 操作时，cuBLAS 内部可能调用 cudaLaunchKernel，
 * 这个标志确保这些内部调用不会被重复拦截。
 */
extern thread_local bool tl_in_scheduler_execution;

} // namespace orion

#endif // ORION_COMMON_H
