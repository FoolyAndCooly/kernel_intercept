/**
 * @file scheduler.h
 * @brief Orion GPU 调度器头文件（简化版）
 *
 * 架构：
 * - 单调度器线程轮询所有客户端队列
 * - HP 直接执行，BE 根据 Orion 逻辑判断
 * - 客户端提交后忙等完成
 */

#ifndef ORION_SCHEDULER_H
#define ORION_SCHEDULER_H

#include "gpu_capture.h"
#include "kernel_profile.h"
#include <cuda_runtime.h>
#include <cuda.h>  // CUDA Driver API for Green Context
#include <cublas_v2.h>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <fstream>
#include <chrono>

namespace orion {

// ============================================================================
// Kernel Profile 信息
// ============================================================================

/**
 * @brief Kernel 性能特征信息
 *
 * 从 kernel_info.csv 加载，用于 Orion 调度决策。
 */
struct KernelProfileInfo {
    std::string name;      // Kernel 名称
    int profile;           // 1: compute-bound, 0: mem-bound, -1: unclear
    int mem;               // 内存占用 (预留)
    int sm_used;           // 需要的 SM 数量
    float duration;        // 执行时间 (ns)

    KernelProfileInfo() : profile(-1), mem(0), sm_used(0), duration(0.0f) {}
    KernelProfileInfo(const std::string& n, int p, int m, int sm, float dur)
        : name(n), profile(p), mem(m), sm_used(sm), duration(dur) {}
};

// ============================================================================
// 执行模式
// ============================================================================

enum class ExecutionMode {
    DEFAULT = 0,   // HP/BE 共享全量 SM，自由竞争
    ISOLATED = 1,  // HP/BE 使用 Green Context SM 隔离
};

// ============================================================================
// 调度配置（简化版）
// ============================================================================

struct SchedulerConfig {
    int device_id = 0;          // GPU 设备 ID
    int num_sms = 0;            // GPU SM 数量（运行时获取）
    int sm_threshold = 0;       // SM 阈值（固定值）
};

// ============================================================================
// Orion 调度状态（简化版）
// ============================================================================

struct OrionSchedulingState {
    // Kernel Profile 信息
    std::vector<std::vector<KernelProfileInfo>> op_info_vector;

    // 每个客户端当前已调度的 kernel 数
    std::vector<int> seen;

    // 每个客户端每次迭代的 kernel 数
    std::vector<int> num_client_kernels;

    // SM 阈值
    int sm_threshold = 0;

    // 保护状态的互斥锁
    std::mutex mutex;

    void init(int num_clients, int num_sms) {
        op_info_vector.resize(num_clients);
        seen.resize(num_clients, 0);
        num_client_kernels.resize(num_clients, 0);
        sm_threshold = num_sms / 2;  // 默认 50%
    }

    void reset() {
        for (auto& v : op_info_vector) v.clear();
        std::fill(seen.begin(), seen.end(), 0);
        std::fill(num_client_kernels.begin(), num_client_kernels.end(), 0);
    }
};

extern OrionSchedulingState g_orion_state;

// ============================================================================
// 调度器类（简化版）
// ============================================================================

class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    // Green Context SM 配置（启动时从环境变量读取）
    struct GreenCtxConfig {
        unsigned int hp_sm_count = 56;
        unsigned int be_sm_count = 56;
        bool enabled = false;
        bool auto_tune = true;
    };

    bool init(int num_clients, const SchedulerConfig& config = SchedulerConfig());
    void start();
    void stop();
    void join();
    void reset();
    int  autotune_green_ctx();  // 在 load_kernel_info 之后、start() 之前调用

    const SchedulerConfig& get_config() const { return config_; }
    SchedulerConfig& get_mutable_config() { return config_; }
    int get_num_clients() const { return num_clients_; }

    bool is_green_ctx_enabled() const { return green_ctx_initialized_; }
    ExecutionMode get_current_mode() const { return current_mode_; }

    // 以下供 C 接口 orion_autotune_green_ctx() 访问
    bool init_green_contexts(const GreenCtxConfig& gc_config);
    GreenCtxConfig choose_best_green_ctx_config_from_profiles() const;

    void sync_client_stream(int client_idx);

    std::atomic<bool> running_{false};

private:
    void run();  // 单线程轮询（legacy，保留用于调试）
    void run_worker(int client_idx);  // 多 worker 模式：每个 client 一个线程
    cudaError_t execute_operation(OperationPtr op, cudaStream_t stream, cublasHandle_t cublas_handle = nullptr);
    bool orion_should_schedule(OperationPtr op, int client_idx);
    bool create_streams();
    void destroy_streams();

    void destroy_green_contexts();
    bool maybe_autotune_static_green_ctx();
    double estimate_client_time_ns(int client_idx, unsigned int allocated_sms) const;

    // Dynamic mode switching
    void switch_mode(ExecutionMode new_mode);
    ExecutionMode decide_mode();
    int peek_sm_requirement(int client_idx, int offset = 0);
    float peek_kernel_duration(int client_idx, int offset = 0);
    bool predict_sm_conflict();

public:
    // Streams (public for C interface access)
    cudaStream_t hp_stream_{nullptr};
    std::vector<cudaStream_t> be_streams_;

private:
    SchedulerConfig config_;
    std::thread thread_;  // legacy 单线程模式
    std::vector<std::thread> workers_;  // 多 worker 模式
    std::atomic<bool> initialized_{false};

    int num_clients_{0};

    // Execution mode state
    ExecutionMode current_mode_ = ExecutionMode::DEFAULT;

    // DEFAULT mode resources (always created)
    cudaStream_t hp_default_stream_ = nullptr;
    std::vector<cudaStream_t> be_default_streams_;

    // ISOLATED mode resources (GC streams, created if GC config is present)
    cudaStream_t hp_gc_stream_ = nullptr;
    std::vector<cudaStream_t> be_gc_streams_;

    // Primary context (saved before GC init, restored on ISOLATED → DEFAULT switch)
    CUcontext primary_ctx_ = nullptr;

    // Cache current GC context index to avoid redundant cuCtxSetCurrent
    int current_ctx_idx_ = -1;

    // Lookahead strategy: operator sequences are fixed and known from
    // kernel_info.csv; seen[i] serves as a program counter to predict
    // future SM conflicts (predict_sm_conflict / decide_mode).
    int lookahead_window_ = 3;

    // Green Context 资源
    CUdevice cu_device_ = 0;
    CUdevResource full_sm_resource_{};
    CUdevResource sm_groups_[2]{};
    CUdevResourceDesc sm_descs_[2] = {nullptr, nullptr};
    CUgreenCtx green_ctxs_[2] = {nullptr, nullptr};    // [0]=HP, [1]=BE
    CUcontext cuda_ctxs_[2] = {nullptr, nullptr};
    cublasHandle_t cublas_handles_[2] = {nullptr, nullptr};  // 每个 context 一个
    bool green_ctx_initialized_ = false;

    GreenCtxConfig gc_config_;
};

extern Scheduler g_scheduler;

bool start_scheduler(int num_clients, const SchedulerConfig& config = SchedulerConfig());
void stop_scheduler();

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_start_scheduler(int num_clients);
void orion_stop_scheduler();

// Kernel profile 加载
int orion_load_kernel_info(int client_idx, const char* file_path);

// 设置客户端 kernel 数
void orion_set_client_kernels(int client_idx, int num_kernels);

// 设置 SM 阈值
void orion_set_sm_threshold(int threshold);

// 获取 SM 阈值
int orion_get_sm_threshold();

// 同步特定客户端的 stream（只等待该客户端的操作完成）
void orion_sync_client_stream(int client_idx);

// 重置状态
void orion_reset_state();

// 搜索最优 Green Context SM 分区并初始化（全程 ISOLATED 模式）
// 必须在 orion_load_kernel_info 之后、调度器线程启动之前调用
// 返回 0 成功，-1 失败
int orion_autotune_green_ctx();

}

#endif // ORION_SCHEDULER_H
