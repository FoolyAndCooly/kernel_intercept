#ifndef ORION_KERNEL_PROFILE_H
#define ORION_KERNEL_PROFILE_H

#include "common.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace orion {

// ============================================================================
// Kernel Profile 信息
// ============================================================================

struct KernelProfile {
    std::string kernel_id;        // 唯一标识 (如 "layer_name:op_index")
    float duration_ms;            // 预期执行时间 (ms)
    int sm_needed;                // 需要的 SM 数量
    ProfileType profile_type;     // compute/memory bound
    
    // 额外信息 (可选)
    int grid_size;                // grid 大小
    int block_size;               // block 大小
    size_t shared_mem_bytes;      // 共享内存使用
    
    KernelProfile() 
        : duration_ms(0.0f)
        , sm_needed(0)
        , profile_type(ProfileType::UNKNOWN)
        , grid_size(0)
        , block_size(0)
        , shared_mem_bytes(0) {}
};

// ============================================================================
// Kernel Profile 表
// ============================================================================

class KernelProfileTable {
public:
    KernelProfileTable();
    ~KernelProfileTable() = default;
    
    // 从 JSON 文件加载 profile
    bool load_from_json(const std::string& filepath);
    
    // 从 YAML 文件加载 profile
    bool load_from_yaml(const std::string& filepath);
    
    // 查找 kernel profile
    const KernelProfile* find(const std::string& kernel_id) const;
    
    // 添加 profile
    void add(const KernelProfile& profile);
    
    // 清空
    void clear();
    
    // 获取所有 profile
    const std::unordered_map<std::string, KernelProfile>& get_all() const {
        return profiles_;
    }
    
    // 计算平均延迟
    float compute_average_duration() const;
    
    // 计算推荐的 DUR_THRESHOLD
    // 基于模型的平均请求延迟
    float compute_recommended_dur_threshold(float target_ratio = 0.025f) const;
    
    // 获取 profile 数量
    size_t size() const { return profiles_.size(); }
    
private:
    std::unordered_map<std::string, KernelProfile> profiles_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Profile 文件格式说明
// ============================================================================

/*
JSON 格式示例:
{
    "model_name": "resnet50",
    "kernels": [
        {
            "kernel_id": "conv1:0",
            "duration_ms": 0.5,
            "sm_needed": 40,
            "profile_type": "compute",
            "grid_size": 1024,
            "block_size": 256
        },
        {
            "kernel_id": "bn1:0",
            "duration_ms": 0.1,
            "sm_needed": 10,
            "profile_type": "memory"
        }
    ]
}

YAML 格式示例:
model_name: resnet50
kernels:
  - kernel_id: conv1:0
    duration_ms: 0.5
    sm_needed: 40
    profile_type: compute
  - kernel_id: bn1:0
    duration_ms: 0.1
    sm_needed: 10
    profile_type: memory
*/

// ============================================================================
// 动态 Profiling 工具
// ============================================================================

class KernelProfiler {
public:
    KernelProfiler();
    ~KernelProfiler();
    
    // 开始 profiling session
    void start_session(const std::string& model_name);
    
    // 结束 session 并输出结果
    void end_session(const std::string& output_path);
    
    // 记录 kernel 执行
    void record_kernel(const std::string& kernel_id, 
                       float duration_ms,
                       int sm_needed,
                       ProfileType profile_type);
    
    // 获取收集的 profile
    void get_profile_table(KernelProfileTable& table) const;
    
private:
    std::string model_name_;
    std::vector<KernelProfile> recorded_profiles_;
    bool session_active_;
    mutable std::mutex mutex_;
};

// 全局 profiler 实例
extern KernelProfiler g_profiler;

} // namespace orion

#endif // ORION_KERNEL_PROFILE_H
