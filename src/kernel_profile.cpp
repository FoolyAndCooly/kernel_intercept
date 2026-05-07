/**
 * @file kernel_profile.cpp
 * @brief Kernel Profile 管理实现
 *
 * 本文件实现了 Orion 调度系统的 Kernel Profile 管理功能，包括：
 * 1. 从 JSON/YAML 文件加载 kernel profile 信息
 * 2. 运行时 kernel 性能记录
 * 3. Profile 数据的查询和统计
 *
 * Kernel Profile 的作用：
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  Kernel Profile 包含以下信息：                                          │
 * ├─────────────────────────────────────────────────────────────────────────┤
 * │  - kernel_id: Kernel 唯一标识符                                         │
 * │  - duration_ms: 执行时间（毫秒）                                        │
 * │  - sm_needed: 需要的 SM 数量                                            │
 * │  - profile_type: compute-bound 或 memory-bound                         │
 * │  - grid_size/block_size: 启动配置（可选）                               │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * 这些信息用于 Orion 调度决策：
 * - SM 需求用于判断是否超过 SM 阈值
 * - Profile 类型用于判断是否可以互补执行
 * - Duration 用于累计时间阈值判断
 *
 * 使用方式：
 * 1. 离线 profiling: 使用 NCU 等工具收集 kernel 性能数据
 * 2. 导出为 JSON/YAML 格式
 * 3. 运行时加载到 KernelProfileTable
 * 4. 调度器查询 profile 信息进行决策
 */

#include "kernel_profile.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace orion {

/**
 * @brief 全局 profiler 实例
 *
 * 用于运行时记录 kernel 性能数据。
 * 可以通过 C 接口 orion_start_profiling/orion_end_profiling 控制。
 */
KernelProfiler g_profiler;

// ============================================================================
// 简单 JSON 解析器 (避免外部依赖)
// ============================================================================
// 实现了一个轻量级的 JSON 解析器，避免引入 nlohmann/json 等外部依赖。
// 只支持我们需要的 JSON 格式，不是完整的 JSON 解析器。

namespace json {

/**
 * @brief 去除字符串两端空白
 *
 * @param s 输入字符串
 * @return 去除空白后的字符串
 */
static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
}

/**
 * @brief 提取引号内的字符串
 *
 * 从当前位置开始，提取双引号包围的字符串内容。
 * 支持转义字符处理。
 *
 * @param s 输入字符串
 * @param pos 当前位置（会被更新到引号后）
 * @return 提取的字符串内容
 */
static std::string extract_string(const std::string& s, size_t& pos) {
    if (pos >= s.length() || s[pos] != '"') return "";
    size_t start = ++pos;
    while (pos < s.length() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.length()) pos++;  // 跳过转义字符
        pos++;
    }
    std::string result = s.substr(start, pos - start);
    if (pos < s.length()) pos++;  // 跳过结束引号
    return result;
}

/**
 * @brief 提取数值
 *
 * 从当前位置开始，提取数值（支持整数、浮点数、科学计数法）。
 *
 * @param s 输入字符串
 * @param pos 当前位置（会被更新到数值后）
 * @return 提取的数值
 */
static double extract_number(const std::string& s, size_t& pos) {
    size_t start = pos;
    while (pos < s.length() && (isdigit(s[pos]) || s[pos] == '.' || s[pos] == '-' || s[pos] == 'e' || s[pos] == 'E' || s[pos] == '+')) {
        pos++;
    }
    return std::stod(s.substr(start, pos - start));
}

/**
 * @brief 跳过空白字符
 *
 * @param s 输入字符串
 * @param pos 当前位置（会被更新）
 */
static void skip_whitespace(const std::string& s, size_t& pos) {
    while (pos < s.length() && isspace(s[pos])) pos++;
}

} // namespace json

// ============================================================================
// KernelProfileTable 实现
// ============================================================================

/**
 * @brief 构造函数
 */
KernelProfileTable::KernelProfileTable() {}

/**
 * @brief 从 JSON 文件加载 kernel profile
 *
 * JSON 文件格式：
 * {
 *     "model_name": "resnet50",
 *     "kernels": [
 *         {
 *             "kernel_id": "conv_fwd_1",
 *             "duration_ms": 0.5,
 *             "sm_needed": 32,
 *             "profile_type": "compute",
 *             "grid_size": 1024,
 *             "block_size": 256
 *         },
 *         ...
 *     ]
 * }
 *
 * @param filepath JSON 文件路径
 * @return 成功返回 true
 */
bool KernelProfileTable::load_from_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open profile file: %s", filepath.c_str());
        return false;
    }
    
    // 读取整个文件
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();
    
    // 简单解析 (不是完整的 JSON 解析器，但足够处理我们的格式)
    size_t pos = 0;
    
    // 查找 "kernels" 数组
    size_t kernels_pos = content.find("\"kernels\"");
    if (kernels_pos == std::string::npos) {
        LOG_ERROR("No 'kernels' array found in profile file");
        return false;
    }
    
    // 找到数组开始的 '['
    pos = content.find('[', kernels_pos);
    if (pos == std::string::npos) {
        LOG_ERROR("Invalid kernels array format");
        return false;
    }
    pos++;  // skip '['
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    while (pos < content.length()) {
        json::skip_whitespace(content, pos);
        
        if (content[pos] == ']') break;  // 数组结束
        if (content[pos] == ',') { pos++; continue; }
        if (content[pos] != '{') { pos++; continue; }
        
        // 解析一个 kernel 对象
        size_t obj_end = content.find('}', pos);
        if (obj_end == std::string::npos) break;
        
        std::string obj_str = content.substr(pos, obj_end - pos + 1);
        pos = obj_end + 1;
        
        KernelProfile profile;
        
        // 提取字段
        size_t field_pos;
        
        // kernel_id
        field_pos = obj_str.find("\"kernel_id\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            field_pos = obj_str.find('"', field_pos);
            if (field_pos != std::string::npos) {
                profile.kernel_id = json::extract_string(obj_str, field_pos);
            }
        }
        
        // duration_ms
        field_pos = obj_str.find("\"duration_ms\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.duration_ms = json::extract_number(obj_str, field_pos);
        }
        
        // sm_needed
        field_pos = obj_str.find("\"sm_needed\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.sm_needed = static_cast<int>(json::extract_number(obj_str, field_pos));
        }
        
        // profile_type
        field_pos = obj_str.find("\"profile_type\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            field_pos = obj_str.find('"', field_pos);
            if (field_pos != std::string::npos) {
                std::string type_str = json::extract_string(obj_str, field_pos);
                if (type_str == "compute") {
                    profile.profile_type = ProfileType::COMPUTE_BOUND;
                } else if (type_str == "memory") {
                    profile.profile_type = ProfileType::MEMORY_BOUND;
                } else {
                    profile.profile_type = ProfileType::UNKNOWN;
                }
            }
        }
        
        // grid_size (可选)
        field_pos = obj_str.find("\"grid_size\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.grid_size = static_cast<int>(json::extract_number(obj_str, field_pos));
        }
        
        // block_size (可选)
        field_pos = obj_str.find("\"block_size\"");
        if (field_pos != std::string::npos) {
            field_pos = obj_str.find(':', field_pos);
            json::skip_whitespace(obj_str, ++field_pos);
            profile.block_size = static_cast<int>(json::extract_number(obj_str, field_pos));
        }
        
        if (!profile.kernel_id.empty()) {
            profiles_[profile.kernel_id] = profile;
            LOG_DEBUG("Loaded profile: %s (%.3f ms, %d SMs, type=%d)",
                      profile.kernel_id.c_str(), profile.duration_ms,
                      profile.sm_needed, (int)profile.profile_type);
        }
    }
    
    LOG_INFO("Loaded %zu kernel profiles from %s", profiles_.size(), filepath.c_str());
    return true;
}

bool KernelProfileTable::load_from_yaml(const std::string& filepath) {
    // YAML 解析类似，这里简化实现
    // 实际应用中可以使用 yaml-cpp 库
    LOG_WARN("YAML parsing not fully implemented, using simplified parser");
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open profile file: %s", filepath.c_str());
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    KernelProfile current;
    bool in_kernels = false;
    std::string line;
    
    while (std::getline(file, line)) {
        line = json::trim(line);
        if (line.empty() || line[0] == '#') continue;
        
        if (line.find("kernels:") != std::string::npos) {
            in_kernels = true;
            continue;
        }
        
        if (in_kernels && line[0] == '-') {
            // 新的 kernel 条目
            if (!current.kernel_id.empty()) {
                profiles_[current.kernel_id] = current;
            }
            current = KernelProfile();
            line = json::trim(line.substr(1));
        }
        
        if (in_kernels) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::string key = json::trim(line.substr(0, colon));
                std::string value = json::trim(line.substr(colon + 1));
                
                if (key == "kernel_id") {
                    current.kernel_id = value;
                } else if (key == "duration_ms") {
                    current.duration_ms = std::stof(value);
                } else if (key == "sm_needed") {
                    current.sm_needed = std::stoi(value);
                } else if (key == "profile_type") {
                    if (value == "compute") {
                        current.profile_type = ProfileType::COMPUTE_BOUND;
                    } else if (value == "memory") {
                        current.profile_type = ProfileType::MEMORY_BOUND;
                    }
                }
            }
        }
    }
    
    if (!current.kernel_id.empty()) {
        profiles_[current.kernel_id] = current;
    }
    
    file.close();
    LOG_INFO("Loaded %zu kernel profiles from %s", profiles_.size(), filepath.c_str());
    return true;
}

const KernelProfile* KernelProfileTable::find(const std::string& kernel_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = profiles_.find(kernel_id);
    return (it != profiles_.end()) ? &it->second : nullptr;
}

void KernelProfileTable::add(const KernelProfile& profile) {
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_[profile.kernel_id] = profile;
}

void KernelProfileTable::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    profiles_.clear();
}

float KernelProfileTable::compute_average_duration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (profiles_.empty()) return 0.0f;
    
    float total = 0.0f;
    for (const auto& pair : profiles_) {
        total += pair.second.duration_ms;
    }
    return total / profiles_.size();
}

float KernelProfileTable::compute_recommended_dur_threshold(float target_ratio) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (profiles_.empty()) return 0.0f;
    
    // 计算总延迟作为一次推理请求的延迟估计
    float total_duration = 0.0f;
    for (const auto& pair : profiles_) {
        total_duration += pair.second.duration_ms;
    }
    
    return total_duration * target_ratio;
}

// ============================================================================
// KernelProfiler 实现
// ============================================================================

KernelProfiler::KernelProfiler() : session_active_(false) {}

KernelProfiler::~KernelProfiler() {
    if (session_active_) {
        end_session("");
    }
}

void KernelProfiler::start_session(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_name_ = model_name;
    recorded_profiles_.clear();
    session_active_ = true;
    LOG_INFO("Started profiling session for model: %s", model_name.c_str());
}

void KernelProfiler::end_session(const std::string& output_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    session_active_ = false;
    
    if (output_path.empty()) {
        LOG_INFO("Profiling session ended, no output file specified");
        return;
    }
    
    // 输出为 JSON
    std::ofstream file(output_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open output file: %s", output_path.c_str());
        return;
    }
    
    file << "{\n";
    file << "    \"model_name\": \"" << model_name_ << "\",\n";
    file << "    \"kernels\": [\n";
    
    for (size_t i = 0; i < recorded_profiles_.size(); i++) {
        const auto& p = recorded_profiles_[i];
        file << "        {\n";
        file << "            \"kernel_id\": \"" << p.kernel_id << "\",\n";
        file << "            \"duration_ms\": " << p.duration_ms << ",\n";
        file << "            \"sm_needed\": " << p.sm_needed << ",\n";
        file << "            \"profile_type\": \"";
        switch (p.profile_type) {
            case ProfileType::COMPUTE_BOUND: file << "compute"; break;
            case ProfileType::MEMORY_BOUND: file << "memory"; break;
            default: file << "unknown"; break;
        }
        file << "\"\n";
        file << "        }";
        if (i < recorded_profiles_.size() - 1) file << ",";
        file << "\n";
    }
    
    file << "    ]\n";
    file << "}\n";
    
    file.close();
    LOG_INFO("Profiling session ended, %zu profiles written to %s", 
             recorded_profiles_.size(), output_path.c_str());
}

void KernelProfiler::record_kernel(const std::string& kernel_id, 
                                    float duration_ms,
                                    int sm_needed,
                                    ProfileType profile_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!session_active_) return;
    
    KernelProfile profile;
    profile.kernel_id = kernel_id;
    profile.duration_ms = duration_ms;
    profile.sm_needed = sm_needed;
    profile.profile_type = profile_type;
    
    recorded_profiles_.push_back(profile);
}

void KernelProfiler::get_profile_table(KernelProfileTable& table) const {
    std::lock_guard<std::mutex> lock(mutex_);
    table.clear();
    for (const auto& p : recorded_profiles_) {
        table.add(p);
    }
}

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_load_profile(const char* filepath) {
    static orion::KernelProfileTable table;
    std::string path(filepath);
    
    if (path.find(".yaml") != std::string::npos || 
        path.find(".yml") != std::string::npos) {
        return table.load_from_yaml(path) ? 0 : -1;
    }
    return table.load_from_json(path) ? 0 : -1;
}

void orion_start_profiling(const char* model_name) {
    orion::g_profiler.start_session(model_name ? model_name : "unnamed");
}

void orion_end_profiling(const char* output_path) {
    orion::g_profiler.end_session(output_path ? output_path : "");
}

} // extern "C"
