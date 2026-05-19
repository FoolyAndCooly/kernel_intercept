/**
 * @file scheduler.cpp
 * @brief Orion GPU 调度器实现（简化版）
 *
 * 架构：
 * - 单调度器线程轮询所有客户端队列
 * - HP (client 0) 直接执行
 * - BE (client 1+) 根据 Orion 逻辑判断：SM 阈值 + Profile 互补
 * - 内存操作直接执行
 * - 没有 HP kernel 在执行时，BE 可以直接执行
 */

#include "scheduler.h"
#include <algorithm>
#include <cstring>

// cuDNN 状态类型（避免引入头文件依赖）
typedef int cudnnStatus_t;

namespace orion {

// ============================================================================
// 全局变量
// ============================================================================

Scheduler g_scheduler;
OrionSchedulingState g_orion_state;

// ============================================================================
// 外部函数声明
// ============================================================================

extern cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cudnnStatus_t execute_cudnn_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cublasStatus_t execute_cublas_operation(OperationPtr op, cudaStream_t scheduler_stream, void* provided_handle = nullptr);
extern cublasStatus_t execute_cublaslt_operation(OperationPtr op, cudaStream_t scheduler_stream);

// 线程局部变量
using orion::tl_is_scheduler_thread;
using orion::tl_worker_idx;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 检查是否是内存操作（直接执行，不走调度判断）
 */
static bool is_memory_operation(OperationType type) {
    return type == OperationType::MALLOC ||
           type == OperationType::FREE ||
           type == OperationType::MEMCPY ||
           type == OperationType::MEMCPY_ASYNC ||
           type == OperationType::MEMSET ||
           type == OperationType::MEMSET_ASYNC;
}

/**
 * @brief 检查是否是需要调度的 kernel 操作
 */
static bool is_kernel_operation(OperationType type) {
    switch (type) {
        case OperationType::KERNEL_LAUNCH:
        case OperationType::KERNEL_LAUNCH_EX:
        case OperationType::KERNEL_LAUNCH_DRV:
        case OperationType::CUDNN_CONV_FWD:
        case OperationType::CUDNN_CONV_BWD_DATA:
        case OperationType::CUDNN_CONV_BWD_FILTER:
        case OperationType::CUDNN_BATCHNORM_FWD:
        case OperationType::CUDNN_BATCHNORM_BWD:
        case OperationType::CUBLAS_SGEMM:
        case OperationType::CUBLAS_SGEMM_BATCHED:
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
        case OperationType::CUBLASLT_MATMUL:
            return true;
        default:
            return false;
    }
}

/**
 * @brief 从 kernel_info.csv 加载 kernel profile 信息
 */
static int populate_kernel_info(int client_idx, const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open kernel_info file: %s", file_path.c_str());
        return -1;
    }

    std::vector<KernelProfileInfo> op_info;
    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        if (line.empty()) continue;

        // 解析 CSV: Name,Profile,Memory_footprint,SM_usage,Duration
        // 从右边解析最后 4 个字段（Name 可能包含逗号）
        std::vector<size_t> comma_positions;
        for (size_t i = 0; i < line.size(); i++) {
            if (line[i] == ',') {
                comma_positions.push_back(i);
            }
        }

        if (comma_positions.size() < 4) {
            LOG_WARN("Invalid CSV line: %s", line.c_str());
            continue;
        }

        size_t n = comma_positions.size();
        size_t pos_profile = comma_positions[n - 4];
        size_t pos_mem = comma_positions[n - 3];
        size_t pos_sm = comma_positions[n - 2];
        size_t pos_dur = comma_positions[n - 1];

        std::string name = line.substr(0, pos_profile);
        std::string profile_str = line.substr(pos_profile + 1, pos_mem - pos_profile - 1);
        std::string mem_str = line.substr(pos_mem + 1, pos_sm - pos_mem - 1);
        std::string sm_str = line.substr(pos_sm + 1, pos_dur - pos_sm - 1);
        std::string dur_str = line.substr(pos_dur + 1);

        KernelProfileInfo info;
        info.name = name;
        try {
            info.profile = std::stoi(profile_str);
            info.mem = std::stoi(mem_str);
            info.sm_used = std::stoi(sm_str);
            info.duration = std::stof(dur_str);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to parse CSV line: %s", line.c_str());
            continue;
        }

        op_info.push_back(info);
    }

    file.close();

    {
        std::lock_guard<std::mutex> lock(g_orion_state.mutex);
        g_orion_state.op_info_vector[client_idx] = std::move(op_info);
    }

    size_t count = g_orion_state.op_info_vector[client_idx].size();
    LOG_INFO("Loaded %zu kernel profiles for client %d from %s", count, client_idx, file_path.c_str());
    return (int)count;
}

// ============================================================================
// Scheduler 实现
// ============================================================================

Scheduler::Scheduler() {}

Scheduler::~Scheduler() {
    stop();
    join();
    destroy_streams();
}

bool Scheduler::init(int num_clients, const SchedulerConfig& config) {
    if (initialized_.load()) {
        LOG_WARN("Scheduler already initialized");
        return true;
    }

    num_clients_ = num_clients;
    config_ = config;

    // 获取 GPU SM 数量
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, config_.device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to get device properties: %s", cudaGetErrorString(err));
        return false;
    }
    config_.num_sms = prop.multiProcessorCount;
    LOG_INFO("GPU has %d SMs", config_.num_sms);

    // 设置默认 SM 阈值
    if (config_.sm_threshold == 0) {
        config_.sm_threshold = config_.num_sms / 2;
    }

    if (!create_streams()) {
        return false;
    }

    // 初始化 Orion 状态
    g_orion_state.init(num_clients, config_.num_sms);
    g_orion_state.sm_threshold = config_.sm_threshold;

    initialized_.store(true);
    LOG_INFO("Scheduler initialized: %d clients, SM threshold=%d", num_clients, config_.sm_threshold);
    return true;
}

// ============================================================================
// Green Context 辅助函数
// ============================================================================

// CUDA Driver API error checking
#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        LOG_ERROR("CUDA Driver API error at %s:%d: %s", __FILE__, __LINE__, errStr); \
        return false; \
    } \
} while(0)

// CUDA Runtime API error checking
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA Runtime API error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

static Scheduler::GreenCtxConfig read_green_ctx_config() {
    Scheduler::GreenCtxConfig gc;
    const char* hp = std::getenv("ORION_HP_SMS");
    const char* be = std::getenv("ORION_BE_SMS");
    if (hp && be) {
        gc.hp_sm_count = std::atoi(hp);
        gc.be_sm_count = std::atoi(be);
        gc.enabled = (gc.hp_sm_count > 0 && gc.be_sm_count > 0);
        if (gc.enabled) {
            LOG_INFO("Green Context config from env: HP=%u SMs, BE=%u SMs",
                     gc.hp_sm_count, gc.be_sm_count);
        }
    }
    return gc;
}

bool Scheduler::init_green_contexts(const GreenCtxConfig& gc_config) {
    gc_config_ = gc_config;

    // Step 1: 初始化 CUDA Driver API（如果还未初始化）
    if (full_sm_resource_.type == CU_DEV_RESOURCE_TYPE_INVALID) {
        CHECK_CU(cuInit(0));
        CHECK_CU(cuDeviceGet(&cu_device_, config_.device_id));

        // Step 1.5: 保存 primary context（在任何 GC 操作之前）
        CHECK_CU(cuDevicePrimaryCtxRetain(&primary_ctx_, cu_device_));

        // Step 2: 获取全部 SM 资源
        CHECK_CU(cuDeviceGetDevResource(
            cu_device_, &full_sm_resource_, CU_DEV_RESOURCE_TYPE_SM));
    }

    // Step 3: 读取 grain（minSmPartitionSize，Hopper 上为 8）并对齐
    const unsigned int grain = full_sm_resource_.sm.minSmPartitionSize;
    const unsigned int align = full_sm_resource_.sm.smCoscheduledAlignment;
    const unsigned int unit  = (align > 0) ? align : (grain > 0 ? grain : 8);

    LOG_INFO("SM resource: total=%u, minPartition=%u, coscheduledAlignment=%u",
             full_sm_resource_.sm.smCount, grain, align);

    if (grain == 0) {
        LOG_ERROR("minSmPartitionSize is 0, cannot partition SMs");
        return false;
    }

    // 向上对齐到 unit 边界
    unsigned int hp_sms = ((gc_config.hp_sm_count + unit - 1) / unit) * unit;
    unsigned int be_sms = ((gc_config.be_sm_count + unit - 1) / unit) * unit;

    // 驱动要求 total <= 120（部分版本），且不能超过 GPU 实际 SM 数
    const unsigned int total_available = full_sm_resource_.sm.smCount;
    const unsigned int MAX_GC_SMS = 120;  // 驱动限制
    if (hp_sms + be_sms > total_available) {
        LOG_ERROR("SM partition exceeds total: HP(%u) + BE(%u) = %u > %u (GPU SMs)",
                  hp_sms, be_sms, hp_sms + be_sms, total_available);
        return false;
    }
    if (hp_sms + be_sms > MAX_GC_SMS) {
        // 等比缩放到 MAX_GC_SMS 以内（保持 unit 对齐）
        float ratio = (float)MAX_GC_SMS / (float)(hp_sms + be_sms);
        unsigned int new_hp = ((unsigned int)(hp_sms * ratio) / unit) * unit;
        unsigned int new_be = ((unsigned int)(be_sms * ratio) / unit) * unit;
        if (new_hp == 0) new_hp = unit;
        if (new_be == 0) new_be = unit;
        // 确保和不超限
        while (new_hp + new_be > MAX_GC_SMS) {
            if (new_be > new_hp) new_be -= unit; else new_hp -= unit;
        }
        LOG_WARN("HP(%u)+BE(%u)=%u > MAX_GC_SMS(%u), scaled to HP=%u BE=%u",
                 hp_sms, be_sms, hp_sms + be_sms, MAX_GC_SMS, new_hp, new_be);
        hp_sms = new_hp;
        be_sms = new_be;
    }

    LOG_INFO("SM partition request: HP=%u SMs, BE=%u SMs (grain=%u)",
             hp_sms, be_sms, grain);

    // Step 4: 非对称分区
    //   用 grain 为粒度切出所有小块，然后将前 chunks_hp 块合并给 HP，
    //   接下来 chunks_be 块合并给 BE（与 test_dual_sm_green_context.cu 相同策略）
    const unsigned int chunk_size = grain;
    const unsigned int chunks_hp  = hp_sms / chunk_size;
    const unsigned int chunks_be  = be_sms / chunk_size;
    const unsigned int total_chunks = chunks_hp + chunks_be;

    // Dry-run：查询驱动实际能创建多少个 chunk
    unsigned int dry_run_count = total_chunks;
    CHECK_CU(cuDevSmResourceSplitByCount(nullptr, &dry_run_count,
                                          &full_sm_resource_, nullptr, 0, chunk_size));

    LOG_INFO("GC split dry-run: chunk_size=%u, need=%u, driver_can=%u",
             chunk_size, total_chunks, dry_run_count);

    if (dry_run_count < total_chunks) {
        LOG_ERROR("Driver can only create %u groups of %u SMs, need %u",
                  dry_run_count, chunk_size, total_chunks);
        return false;
    }

    // 实际切分
    std::vector<CUdevResource> chunks(dry_run_count);
    CUdevResource remaining{};
    unsigned int group_count = dry_run_count;
    CHECK_CU(cuDevSmResourceSplitByCount(chunks.data(), &group_count,
                                          &full_sm_resource_, &remaining, 0, chunk_size));

    LOG_INFO("SM split: %u chunks x %u SMs each, remaining %u SMs",
             group_count, chunks[0].sm.smCount, remaining.sm.smCount);

    // Step 5: 合并 chunks 并创建 Green Context
    //   GC[0]=HP: chunks[0 .. chunks_hp-1]
    //   GC[1]=BE: chunks[chunks_hp .. chunks_hp+chunks_be-1]
    CUdevResourceDesc desc_hp = nullptr;
    CHECK_CU(cuDevResourceGenerateDesc(&desc_hp, &chunks[0], chunks_hp));
    CHECK_CU(cuGreenCtxCreate(&green_ctxs_[0], desc_hp,
                               cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuCtxFromGreenCtx(&cuda_ctxs_[0], green_ctxs_[0]));
    sm_descs_[0] = desc_hp;

    unsigned int actual_hp = 0;
    for (unsigned int i = 0; i < chunks_hp; i++) actual_hp += chunks[i].sm.smCount;

    CUdevResourceDesc desc_be = nullptr;
    CHECK_CU(cuDevResourceGenerateDesc(&desc_be, &chunks[chunks_hp], chunks_be));
    CHECK_CU(cuGreenCtxCreate(&green_ctxs_[1], desc_be,
                               cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuCtxFromGreenCtx(&cuda_ctxs_[1], green_ctxs_[1]));
    sm_descs_[1] = desc_be;

    unsigned int actual_be = 0;
    for (unsigned int i = chunks_hp; i < chunks_hp + chunks_be; i++)
        actual_be += chunks[i].sm.smCount;

    LOG_INFO("Green Context created: HP=%u SMs, BE=%u SMs", actual_hp, actual_be);

    // Step 6: 在各自 Green Context 中创建 Stream 和 cuBLAS Handle
    // HP GC stream
    CHECK_CU(cuCtxSetCurrent(cuda_ctxs_[0]));
    CHECK_CUDA(cudaSetDevice(config_.device_id));
    CUstream cu_hp_stream;
    CHECK_CU(cuGreenCtxStreamCreate(&cu_hp_stream, green_ctxs_[0],
                                     CU_STREAM_NON_BLOCKING, 0));
    hp_gc_stream_ = reinterpret_cast<cudaStream_t>(cu_hp_stream);

    cublasStatus_t cublas_err = cublasCreate(&cublas_handles_[0]);
    if (cublas_err != 0) {
        LOG_ERROR("Failed to create HP cuBLAS handle: %d", cublas_err);
        return false;
    }
    LOG_INFO("HP Green Context: %u SMs, stream + cuBLAS handle created", actual_hp);

    // BE GC stream
    CHECK_CU(cuCtxSetCurrent(cuda_ctxs_[1]));
    CHECK_CUDA(cudaSetDevice(config_.device_id));
    CUstream cu_be_stream;
    CHECK_CU(cuGreenCtxStreamCreate(&cu_be_stream, green_ctxs_[1],
                                     CU_STREAM_NON_BLOCKING, 0));
    be_gc_streams_.resize(1);
    be_gc_streams_[0] = reinterpret_cast<cudaStream_t>(cu_be_stream);

    cublas_err = cublasCreate(&cublas_handles_[1]);
    if (cublas_err != 0) {
        LOG_ERROR("Failed to create BE cuBLAS handle: %d", cublas_err);
        return false;
    }
    LOG_INFO("BE Green Context: %u SMs, stream + cuBLAS handle created", actual_be);

    // Step A: dummy prime —— 在每个 GC context 下预跑一次小 kernel，
    // 提前触发 CUDA module loader / JIT cache / cuBLAS workspace 分配，
    // 避免首个真实 workload op 把冷启动开销计入测时段。
    auto prime_ctx = [&](int ctx_idx, cudaStream_t s) {
        if (cuCtxSetCurrent(cuda_ctxs_[ctx_idx]) != CUDA_SUCCESS) return;

        // 分配 device 内存用于 dummy 操作
        float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        if (cudaMalloc(&d_a, sizeof(float)) != cudaSuccess) return;
        if (cudaMalloc(&d_b, sizeof(float)) != cudaSuccess) {
            cudaFree(d_a);
            return;
        }
        if (cudaMalloc(&d_c, sizeof(float)) != cudaSuccess) {
            cudaFree(d_a);
            cudaFree(d_b);
            return;
        }

        // 初始化为 0
        cudaMemsetAsync(d_a, 0, sizeof(float), s);
        cudaMemsetAsync(d_b, 0, sizeof(float), s);
        cudaMemsetAsync(d_c, 0, sizeof(float), s);

        // 触发 cuBLAS 内部 workspace 初始化：1x1 的 sgemm
        cublasHandle_t h = cublas_handles_[ctx_idx];
        if (h) {
            cublasSetStream(h, s);
            float alpha = 1.f, beta = 0.f;
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1,
                        &alpha, d_a, 1, d_b, 1, &beta, d_c, 1);
        }

        cudaStreamSynchronize(s);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    };
    prime_ctx(0, hp_gc_stream_);
    prime_ctx(1, be_gc_streams_[0]);

    // Step 7: 注册 GC stream 到 client 映射表，让 cuBLAS/cuDNN 内部线程
    // 通过 stream 指针也能 resolve 到正确的 client
    register_client_stream(0, (void*)hp_gc_stream_);
    register_client_stream(1, (void*)be_gc_streams_[0]);

    // Step 8: 恢复 primary context（启动时默认 ISOLATED 模式直接用 GC）
    CHECK_CU(cuCtxSetCurrent(primary_ctx_));

    green_ctx_initialized_ = true;
    return true;
}

void Scheduler::destroy_green_contexts() {
    if (!green_ctx_initialized_) return;

    // Destroy cuBLAS handles
    for (int i = 0; i < 2; i++) {
        if (cublas_handles_[i]) {
            cublasDestroy(cublas_handles_[i]);
            cublas_handles_[i] = nullptr;
        }
    }

    // Destroy Green Contexts (streams are destroyed with contexts)
    for (int i = 0; i < 2; i++) {
        if (green_ctxs_[i]) {
            cuGreenCtxDestroy(green_ctxs_[i]);
            green_ctxs_[i] = nullptr;
        }
        // Note: CUdevResourceDesc is opaque and managed by CUDA driver
        sm_descs_[i] = nullptr;
    }

    green_ctx_initialized_ = false;
    LOG_INFO("Green Context resources destroyed");
}

// ============================================================================
// 动态模式切换（GPU 执行预测 + SM 冲突检测 + 冷却防抖）
// ============================================================================

void Scheduler::switch_mode(ExecutionMode new_mode) {
    if (new_mode == current_mode_) return;

    LOG_INFO("[MODE-SWITCH] ===== Mode transition: %s -> %s =====",
             current_mode_ == ExecutionMode::DEFAULT ? "DEFAULT" : "ISOLATED",
             new_mode == ExecutionMode::DEFAULT ? "DEFAULT" : "ISOLATED");

    // 等待当前模式所有 stream 上的 kernel 完成
    if (current_mode_ == ExecutionMode::DEFAULT) {
        LOG_DEBUG("[MODE-SWITCH] Synchronizing DEFAULT streams before transition...");
        if (hp_default_stream_) cudaStreamSynchronize(hp_default_stream_);
        for (auto s : be_default_streams_) {
            if (s) cudaStreamSynchronize(s);
        }
    } else {
        LOG_DEBUG("[MODE-SWITCH] Synchronizing ISOLATED (GC) streams before transition...");
        if (hp_gc_stream_) cudaStreamSynchronize(hp_gc_stream_);
        for (auto s : be_gc_streams_) {
            if (s) cudaStreamSynchronize(s);
        }
        LOG_DEBUG("[MODE-SWITCH] Restoring primary context (ISOLATED -> DEFAULT)");
        cuCtxSetCurrent(primary_ctx_);
        current_ctx_idx_ = -1;
    }

    current_mode_ = new_mode;

    // 更新 hp_stream_/be_streams_ 别名指向当前模式资源
    if (current_mode_ == ExecutionMode::DEFAULT) {
        hp_stream_ = hp_default_stream_;
        for (int i = 0; i < (int)be_default_streams_.size(); i++) {
            be_streams_[i] = be_default_streams_[i];
        }
        LOG_DEBUG("[MODE-SWITCH] Stream aliases updated to DEFAULT resources");
    } else {
        hp_stream_ = hp_gc_stream_;
        for (auto& s : be_streams_) {
            s = (!be_gc_streams_.empty()) ? be_gc_streams_[0] : nullptr;
        }
        LOG_DEBUG("[MODE-SWITCH] Stream aliases updated to ISOLATED (GC) resources");
    }

    LOG_INFO("[MODE-SWITCH] ===== Now running in %s mode =====",
             current_mode_ == ExecutionMode::DEFAULT ? "DEFAULT" : "ISOLATED");
}

int Scheduler::peek_sm_requirement(int client_idx, int offset) {
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);

    if (client_idx < 0 || client_idx >= (int)g_orion_state.op_info_vector.size())
        return 0;

    auto& profile = g_orion_state.op_info_vector[client_idx];
    if (profile.empty()) return 0;

    int n = (int)profile.size();
    int seen = g_orion_state.seen[client_idx];
    int idx = ((seen + offset) % n + n) % n;
    return profile[idx].sm_used;
}

float Scheduler::peek_kernel_duration(int client_idx, int offset) {
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);

    if (client_idx < 0 || client_idx >= (int)g_orion_state.op_info_vector.size())
        return 0.0f;

    auto& profile = g_orion_state.op_info_vector[client_idx];
    if (profile.empty()) return 0.0f;

    int n = (int)profile.size();
    int seen = g_orion_state.seen[client_idx];
    int idx = ((seen + offset) % n + n) % n;
    return profile[idx].duration;
}

/**
 * @brief 前瞻式 SM 冲突预测（max-in-window 策略）
 *
 * 利用算子序列固定且完全可预测的特性，通过 seen[i] 程序计数器
 * 向前扫描未来 lookahead_window_ 步的 SM 需求，判断是否存在冲突。
 *
 * 活跃客户端判定：基于 profile 数据（seen[i] > 0 且 profile 非空），
 * 而非队列头部的瞬时状态。队列头部检测不可靠——调度器每秒轮询
 * 百万次但 op 只有数百个，kernel op 出现在队列头部的窗口极窄，
 * 且被 MALLOC/MEMCPY 等非 kernel op 频繁打断。profile 数据
 * 始终可用，一旦客户端开始执行（seen > 0），其未来序列完全确定。
 *
 * SM 比较策略：取每个客户端在窗口内的 **最大 SM 需求** 再求和
 * （max-in-window）。原因：长短 kernel 的时长差异导致不同 step
 * 的 kernel 会在 GPU 上时间重叠。
 */
bool Scheduler::predict_sm_conflict() {
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);

    int active_count = 0;
    for (int i = 0; i < num_clients_; i++) {
        if (i < (int)g_orion_state.op_info_vector.size() &&
            !g_orion_state.op_info_vector[i].empty() &&
            g_orion_state.seen[i] > 0) {
            active_count++;
        }
    }
    if (active_count < 2) return false;

    int total_max_sm = 0;
    for (int i = 0; i < num_clients_; i++) {
        auto& profile = g_orion_state.op_info_vector[i];
        if (profile.empty()) continue;
        int n = (int)profile.size();
        int seen = g_orion_state.seen[i];
        int max_sm = 0;
        for (int step = 0; step <= lookahead_window_; step++) {
            int idx = ((seen + step) % n + n) % n;
            max_sm = std::max(max_sm, profile[idx].sm_used);
        }
        total_max_sm += max_sm;
    }

    if (total_max_sm > config_.num_sms) {
        LOG_DEBUG("[LOOKAHEAD] SM conflict predicted (max-in-window): total_max_sm=%d > %d",
                 total_max_sm, config_.num_sms);
        return true;
    }

    return false;
}

// ============================================================================
// 静态 Green Context 分区搜索
// ============================================================================

/**
 * 估算某个客户端在分配 allocated_sms 个 SM 时，跑完一轮所有算子的总时间（ns）。
 *
 * Wave 模型（修正版）：
 *   1. op.sm_used 是理论 SM 需求（可能 > GPU 总 SM 数）
 *   2. NCU profiling 时的 wave 数 = ceil(op.sm_used / num_sms)
 *   3. 单 wave 时长 = op.duration / waves_at_profiling
 *   4. 新分配下的 wave 数 = ceil(op.sm_used / allocated_sms)
 *   5. 估算时长 = single_wave_duration × waves_at_new_allocation
 *
 * 这样可以正确处理 op.sm_used > num_sms 的情况（duration 已包含多个 wave）。
 */
double Scheduler::estimate_client_time_ns(int client_idx, unsigned int allocated_sms) const {
    if (client_idx >= (int)g_orion_state.op_info_vector.size()) return 0.0;
    const auto& ops = g_orion_state.op_info_vector[client_idx];
    if (ops.empty() || allocated_sms == 0) return 0.0;

    double total = 0.0;
    for (const auto& op : ops) {
        int sm_theoretical = op.sm_used > 0 ? op.sm_used : 1;

        // NCU profiling 时的 wave 数（GPU 总 SM 数）
        int waves_at_profiling = (sm_theoretical + config_.num_sms - 1) / config_.num_sms;

        // 计算单 wave 时长
        double single_wave_duration = (double)op.duration / (double)waves_at_profiling;

        // 新分配下的 wave 数
        int waves_at_new_allocation = (sm_theoretical + (int)allocated_sms - 1) / (int)allocated_sms;

        // 估算总时长
        double estimated_duration = single_wave_duration * (double)waves_at_new_allocation;

        total += estimated_duration;
    }
    return total;
}

/**
 * 枚举所有合法的 (hp_sms, be_sms) 分区，选出使 max(T_hp, T_be) 最小的配置。
 *
 * 约束（与 init_green_contexts 保持一致）：
 *   - 每组必须是 grain（8）的倍数
 *   - hp_sms + be_sms <= min(total_sm, 120)
 *   - 每组 >= grain
 */
Scheduler::GreenCtxConfig Scheduler::choose_best_green_ctx_config_from_profiles() const {
    const unsigned int grain = 8;
    const unsigned int total_sm = (unsigned int)config_.num_sms;
    const unsigned int cap = total_sm;  // 使用实际 SM 数，不做人为截断

    // 查询 Driver 实际能创建的最大 chunk 数量
    unsigned int max_driver_chunks = 0;
    if (full_sm_resource_.type != CU_DEV_RESOURCE_TYPE_INVALID) {
        unsigned int probe = cap / grain;
        cuDevSmResourceSplitByCount(nullptr, &probe, &full_sm_resource_, nullptr, 0, grain);
        max_driver_chunks = probe;
    }
    if (max_driver_chunks == 0) {
        // 无法查询时保守估计：总 SM / grain - 1
        max_driver_chunks = cap / grain - 1;
    }
    LOG_INFO("[GC-SEARCH] Driver max chunks: %u (grain=%u)", max_driver_chunks, grain);

    double best_time = std::numeric_limits<double>::max();
    unsigned int best_hp = 0, best_be = 0;

    for (unsigned int hp = grain; hp <= max_driver_chunks * grain; hp += grain) {
        unsigned int remaining_chunks = max_driver_chunks - hp / grain;
        if (remaining_chunks == 0) continue;
        // be 取剩余 chunk 数对应的最大 SM 数
        unsigned int be = remaining_chunks * grain;
        if (be < grain) continue;

        double t_hp = estimate_client_time_ns(0, hp);
        double t_be = estimate_client_time_ns(1, be);
        double t_total = std::max(t_hp, t_be);

        LOG_DEBUG("[GC-SEARCH] HP=%u BE=%u -> T_hp=%.0fns T_be=%.0fns max=%.0fns",
                  hp, be, t_hp, t_be, t_total);

        if (t_total < best_time) {
            best_time = t_total;
            best_hp = hp;
            best_be = be;
        }
    }

    GreenCtxConfig cfg;
    if (best_hp > 0 && best_be > 0) {
        cfg.hp_sm_count = best_hp;
        cfg.be_sm_count = best_be;
        cfg.enabled = true;
        LOG_INFO("[GC-SEARCH] Best partition: HP=%u SMs, BE=%u SMs, est_max=%.3f ms",
                 best_hp, best_be, best_time / 1e6);
    } else {
        // profile 数据不足或无满足约束的分区，回退到均分（不超过 driver 限制）
        unsigned int half = (max_driver_chunks / 2) * grain;
        if (half < grain) half = grain;
        cfg.hp_sm_count = half;
        cfg.be_sm_count = half;
        cfg.enabled = true;
        LOG_WARN("[GC-SEARCH] No valid partition found, fallback to HP=%u BE=%u", half, half);
    }
    return cfg;
}

/**
 * 若 ORION_GC_AUTOTUNE=1 且两个客户端都已加载 profile，
 * 则自动搜索最优分区并覆盖环境变量配置。
 * 返回 true 表示已完成搜索并更新了 gc_config_。
 */
bool Scheduler::maybe_autotune_static_green_ctx() {
    const char* autotune = std::getenv("ORION_GC_AUTOTUNE");
    if (!autotune || std::atoi(autotune) != 1) return false;

    bool hp_ready = !g_orion_state.op_info_vector.empty() &&
                    !g_orion_state.op_info_vector[0].empty();
    bool be_ready = (int)g_orion_state.op_info_vector.size() > 1 &&
                    !g_orion_state.op_info_vector[1].empty();

    if (!hp_ready || !be_ready) {
        LOG_WARN("[GC-SEARCH] ORION_GC_AUTOTUNE=1 but profiles not loaded yet, skipping");
        return false;
    }

    gc_config_ = choose_best_green_ctx_config_from_profiles();
    return gc_config_.enabled;
}

int Scheduler::autotune_green_ctx() {
    if (!initialized_.load()) {
        LOG_ERROR("autotune_green_ctx: scheduler not initialized");
        return -1;
    }
    if (running_.load()) {
        LOG_ERROR("autotune_green_ctx: scheduler already running, call before start()");
        return -1;
    }

    bool hp_ready = !g_orion_state.op_info_vector.empty() &&
                    !g_orion_state.op_info_vector[0].empty();
    bool be_ready = (int)g_orion_state.op_info_vector.size() > 1 &&
                    !g_orion_state.op_info_vector[1].empty();
    if (!hp_ready || !be_ready) {
        LOG_ERROR("autotune_green_ctx: profiles not loaded (call orion_load_kernel_info first)");
        return -1;
    }

    // 初始化 CUDA Driver 和获取 SM 资源（用于搜索时查询 Driver 限制）
    if (full_sm_resource_.type == CU_DEV_RESOURCE_TYPE_INVALID) {
        CHECK_CU(cuInit(0));
        CHECK_CU(cuDeviceGet(&cu_device_, config_.device_id));
        CHECK_CU(cuDevicePrimaryCtxRetain(&primary_ctx_, cu_device_));
        CHECK_CU(cuDeviceGetDevResource(cu_device_, &full_sm_resource_, CU_DEV_RESOURCE_TYPE_SM));
        LOG_INFO("SM resource initialized for autotune: total=%u SMs", full_sm_resource_.sm.smCount);
    }

    GreenCtxConfig best = choose_best_green_ctx_config_from_profiles();
    if (!best.enabled) {
        LOG_ERROR("autotune_green_ctx: partition search failed");
        return -1;
    }

    if (green_ctx_initialized_) {
        LOG_WARN("autotune_green_ctx: reinitializing GC with new partition");
        destroy_green_contexts();
        hp_gc_stream_ = nullptr;
        be_gc_streams_.clear();
    }

    if (!init_green_contexts(best)) {
        LOG_ERROR("autotune_green_ctx: Green Context init failed");
        return -1;
    }

    // 全程 ISOLATED
    current_mode_ = ExecutionMode::ISOLATED;
    hp_stream_ = hp_gc_stream_;
    for (auto& s : be_streams_) {
        s = be_gc_streams_.empty() ? nullptr : be_gc_streams_[0];
    }

    LOG_INFO("autotune_green_ctx done: HP=%u SMs, BE=%u SMs, mode=ISOLATED",
             best.hp_sm_count, best.be_sm_count);
    return 0;
}

void Scheduler::sync_client_stream(int client_idx) {
    // 同步该客户端在 DEFAULT 和 ISOLATED 两条轨道上可能存在的工作
    if (client_idx == 0) {
        if (hp_default_stream_) cudaStreamSynchronize(hp_default_stream_);
        if (hp_gc_stream_) cudaStreamSynchronize(hp_gc_stream_);
    } else {
        int be_idx = client_idx - 1;
        if (be_idx < (int)be_default_streams_.size() && be_default_streams_[be_idx])
            cudaStreamSynchronize(be_default_streams_[be_idx]);
        if (!be_gc_streams_.empty() && be_gc_streams_[0])
            cudaStreamSynchronize(be_gc_streams_[0]);
    }
}

bool Scheduler::create_streams() {
    // ========================================================================
    // Phase 1: 始终创建 DEFAULT 模式资源（stream priority 方案）
    // ========================================================================
    int lowest_priority, highest_priority;
    cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);

    LOG_INFO("Stream priority range: [%d (highest), %d (lowest)]", highest_priority, lowest_priority);

    // HP DEFAULT stream (最高优先级)
    cudaError_t err = cudaStreamCreateWithPriority(&hp_default_stream_, cudaStreamNonBlocking, highest_priority);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to create HP default stream: %s", cudaGetErrorString(err));
        return false;
    }
    LOG_INFO("HP default stream created with priority %d", highest_priority);

    // BE DEFAULT streams (优先级递减)
    int num_be = num_clients_ - 1;
    be_default_streams_.resize(num_be);
    int priority_range = lowest_priority - highest_priority;

    for (int i = 0; i < num_be; i++) {
        int be_priority;
        if (priority_range > 0 && num_be > 0) {
            be_priority = highest_priority + 1 + (i * (priority_range - 1)) / std::max(1, num_be - 1);
            be_priority = std::min(be_priority, lowest_priority);
        } else {
            be_priority = lowest_priority;
        }

        err = cudaStreamCreateWithPriority(&be_default_streams_[i], cudaStreamNonBlocking, be_priority);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create BE%d default stream: %s", i + 1, cudaGetErrorString(err));
            return false;
        }
        LOG_INFO("BE%d default stream created with priority %d", i + 1, be_priority);
    }

    // 初始状态：hp_stream_/be_streams_ 指向 DEFAULT 资源
    hp_stream_ = hp_default_stream_;
    be_streams_ = be_default_streams_;
    current_mode_ = ExecutionMode::DEFAULT;

    LOG_INFO("DEFAULT mode resources created: 1 HP + %d BE streams", num_be);

    // ========================================================================
    // Phase 2: 创建 Green Context（全程 ISOLATED）
    //
    // 分区来源优先级：
    //   1) ORION_HP_SMS / ORION_BE_SMS 环境变量 -> 手动指定，立即初始化
    //   2) ORION_GC_AUTOTUNE=1                  -> 等 profile 加载后由
    //                                              orion_autotune_green_ctx() 触发
    //   3) 两者都没有                            -> 跳过，保持 DEFAULT
    // ========================================================================
    GreenCtxConfig gc_config = read_green_ctx_config();

    if (!gc_config.enabled) {
        // 没有手动指定分区；若设了 AUTOTUNE，等 profile 加载后再初始化
        const char* autotune = std::getenv("ORION_GC_AUTOTUNE");
        if (autotune && std::atoi(autotune) == 1) {
            LOG_INFO("ORION_GC_AUTOTUNE=1: GC init deferred until orion_autotune_green_ctx() is called");
        } else {
            LOG_INFO("Green Context not configured, running in DEFAULT mode");
        }
        return true;
    }

    if (!init_green_contexts(gc_config)) {
        LOG_WARN("Green Context init failed, falling back to DEFAULT mode");
        return true;
    }

    // 全程 ISOLATED：直接切到 GC stream
    current_mode_ = ExecutionMode::ISOLATED;
    hp_stream_ = hp_gc_stream_;
    for (auto& s : be_streams_) {
        s = (!be_gc_streams_.empty()) ? be_gc_streams_[0] : nullptr;
    }
    LOG_INFO("Running in static ISOLATED (Green Context) mode: HP=%u SMs, BE=%u SMs",
             gc_config.hp_sm_count, gc_config.be_sm_count);

    return true;
}

void Scheduler::destroy_streams() {
    // 清理 ISOLATED 资源（GC streams 随 Green Context 一起销毁）
    if (green_ctx_initialized_) {
        destroy_green_contexts();
    }
    hp_gc_stream_ = nullptr;
    be_gc_streams_.clear();

    // 清理 DEFAULT 资源
    if (hp_default_stream_) {
        cudaStreamDestroy(hp_default_stream_);
        hp_default_stream_ = nullptr;
    }
    for (auto& stream : be_default_streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    be_default_streams_.clear();

    // 释放 primary context
    if (primary_ctx_) {
        cuDevicePrimaryCtxRelease(cu_device_);
        primary_ctx_ = nullptr;
    }

    // 清理别名
    hp_stream_ = nullptr;
    be_streams_.clear();

    current_mode_ = ExecutionMode::DEFAULT;
    current_ctx_idx_ = -1;
}

void Scheduler::start() {
    if (!initialized_.load()) {
        LOG_ERROR("Scheduler not initialized");
        return;
    }
    if (running_.load()) {
        LOG_WARN("Scheduler already running");
        return;
    }

    running_.store(true);

    // 多 worker 模式：每个 client 一个独立线程
    workers_.resize(num_clients_);
    for (int i = 0; i < num_clients_; i++) {
        workers_[i] = std::thread(&Scheduler::run_worker, this, i);
    }
    LOG_INFO("Scheduler started with %d worker threads (multi-worker mode)", num_clients_);
}

void Scheduler::stop() {
    if (!running_.load()) return;

    LOG_INFO("Stopping scheduler...");
    running_.store(false);

    // 关闭所有队列
    for (int i = 0; i < num_clients_; i++) {
        if (g_capture_state.client_queues[i]) {
            g_capture_state.client_queues[i]->shutdown();
        }
    }
}

void Scheduler::join() {
    // 多 worker 模式
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();

    // 兼容 legacy 单线程模式
    if (thread_.joinable()) {
        thread_.join();
    }
    LOG_INFO("All scheduler workers joined");
}

void Scheduler::reset() {
    if (running_.load()) {
        LOG_WARN("Cannot reset while running");
        return;
    }
    destroy_streams();
    initialized_.store(false);
    num_clients_ = 0;
    g_orion_state.reset();
    LOG_INFO("Scheduler reset");
}

/**
 * @brief Orion 调度判断：BE kernel 是否可以执行
 *
 * ISOLATED 模式：HP 和 BE 使用独立 SM 分区，始终允许并发执行。
 * DEFAULT 模式（并发）：检查 BE kernel 与 HP 当前 GPU 工作的 SM 总需求，
 *   若不冲突（总和 ≤ GPU SM 数）则允许并发，否则等 HP 空闲。
 */
bool Scheduler::orion_should_schedule(OperationPtr op, int client_idx) {
    if (is_memory_operation(op->type)) {
        LOG_INFO("[SHOULD_SCHEDULE] Client %d: memory op, allowed", client_idx);
        return true;
    }
    if (!is_kernel_operation(op->type)) {
        LOG_INFO("[SHOULD_SCHEDULE] Client %d: non-kernel op, allowed", client_idx);
        return true;
    }

    // ISOLATED 模式：HP/BE 各有独立 SM 分区，无需互相等待
    if (current_mode_ == ExecutionMode::ISOLATED) {
        LOG_INFO("[SHOULD_SCHEDULE] Client %d: BE kernel allowed (ISOLATED mode)", client_idx);
        return true;
    }

    // DEFAULT 模式：利用 profile 数据直接检查 SM 兼容性
    int hp_sm = peek_sm_requirement(0);
    int be_sm = peek_sm_requirement(client_idx);
    int total_sm = hp_sm + be_sm;

    if (total_sm <= config_.num_sms) {
        LOG_DEBUG("[SCHEDULE] Client %d: BE allowed (DEFAULT, HP_sm=%d + BE_sm=%d = %d <= %d)",
                 client_idx, hp_sm, be_sm, total_sm, config_.num_sms);
        return true;
    }

    // HP 队列为空，无竞争
    if (g_capture_state.client_queues[0] &&
        g_capture_state.client_queues[0]->empty()) {
        LOG_DEBUG("[SCHEDULE] Client %d: BE allowed (DEFAULT, HP queue empty)", client_idx);
        return true;
    }

    LOG_DEBUG("[SCHEDULE] Client %d: BE BLOCKED (DEFAULT, SM conflict: HP=%d + BE=%d = %d > %d)",
             client_idx, hp_sm, be_sm, total_sm, config_.num_sms);
    return false;
}

/**
 * @brief 执行操作
 */
cudaError_t Scheduler::execute_operation(OperationPtr op, cudaStream_t stream, cublasHandle_t cublas_handle) {
    op->started.store(true);

    switch (op->type) {
        case OperationType::KERNEL_LAUNCH:
        case OperationType::KERNEL_LAUNCH_EX:
        case OperationType::KERNEL_LAUNCH_DRV:
        case OperationType::MALLOC:
        case OperationType::FREE:
        case OperationType::MEMCPY:
        case OperationType::MEMCPY_ASYNC:
        case OperationType::MEMSET:
        case OperationType::MEMSET_ASYNC:
        case OperationType::DEVICE_SYNC:
        case OperationType::STREAM_SYNC:
            return execute_cuda_operation(op, stream);

        case OperationType::CUDNN_CONV_FWD:
        case OperationType::CUDNN_CONV_BWD_DATA:
        case OperationType::CUDNN_CONV_BWD_FILTER:
        case OperationType::CUDNN_BATCHNORM_FWD:
        case OperationType::CUDNN_BATCHNORM_BWD:
            return execute_cudnn_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        case OperationType::CUBLAS_SGEMM:
        case OperationType::CUBLAS_SGEMM_BATCHED:
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
            // 传递 cuBLAS handle（如果有），转换为 void*
            return execute_cublas_operation(op, stream, (void*)cublas_handle) == 0 ? cudaSuccess : cudaErrorUnknown;

        case OperationType::CUBLASLT_MATMUL:
            return execute_cublaslt_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        default:
            LOG_ERROR("Unknown operation type: %d", (int)op->type);
            return cudaErrorUnknown;
    }
}

/**
 * @brief 多 Worker 模式：每个 client 一个独立线程
 *
 * 每个 worker 独立消费自己的 ClientQueue，天然并行提交 kernel。
 * HP worker 直接执行，BE worker 执行前检查 orion_should_schedule。
 * Green Context 模式下，每个 worker 在自己的线程中设置对应的 CUcontext。
 */
void Scheduler::run_worker(int client_idx) {
    tl_is_scheduler_thread = true;
    tl_worker_idx = client_idx;
    LOG_INFO("Worker %d started (multi-worker mode)", client_idx);

    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Worker %d: cudaSetDevice(%d) failed: %s",
                  client_idx, config_.device_id, cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_WARN("Worker %d: cudaDeviceSynchronize returned %d: %s",
                  client_idx, err, cudaGetErrorString(err));
    }

    // Green Context 模式：在 worker 线程中设置对应的 CUcontext
    cublasHandle_t worker_cublas_handle = nullptr;
    cudaStream_t worker_stream = nullptr;

    if (current_mode_ == ExecutionMode::ISOLATED && green_ctx_initialized_) {
        int ctx_idx = (client_idx == 0) ? 0 : 1;
        cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
        worker_cublas_handle = cublas_handles_[ctx_idx];
        worker_stream = (client_idx == 0) ? hp_gc_stream_ : be_gc_streams_[0];
        LOG_INFO("Worker %d: using Green Context %d, stream=%p", client_idx, ctx_idx, worker_stream);
    } else {
        worker_stream = (client_idx == 0) ? hp_stream_ : be_streams_[client_idx - 1];
        LOG_INFO("Worker %d: using DEFAULT stream=%p", client_idx, worker_stream);
    }

    auto* q = g_capture_state.client_queues[client_idx].get();
    if (!q) {
        LOG_ERROR("Worker %d: queue is null", client_idx);
        return;
    }

    while (running_.load()) {
        auto op = q->try_pop();
        if (!op) {
            std::this_thread::yield();
            continue;
        }

        // BE worker：检查调度判断
        if (client_idx > 0 && !orion_should_schedule(op, client_idx)) {
            // 放回队列头部不可行（queue 是 FIFO），改为忙等
            // 重新 push 回去并 yield
            q->push(op);
            std::this_thread::yield();
            continue;
        }

        // 选择执行 stream：优先使用 op 自带的 stream
        cudaStream_t exec_stream = worker_stream;
        switch (op->type) {
            case OperationType::KERNEL_LAUNCH:
            case OperationType::KERNEL_LAUNCH_EX:
            case OperationType::KERNEL_LAUNCH_DRV: {
                auto& p = std::get<KernelLaunchParams>(op->params);
                if (p.stream) exec_stream = p.stream;
                break;
            }
            case OperationType::CUBLAS_SGEMM:
            case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
                auto& p = std::get<CublasGemmParams>(op->params);
                if (p.stream) exec_stream = p.stream;
                break;
            }
            case OperationType::CUBLASLT_MATMUL: {
                auto& p = std::get<CublasLtMatmulParams>(op->params);
                if (p.stream) exec_stream = p.stream;
                break;
            }
            case OperationType::MEMCPY_ASYNC: {
                auto& p = std::get<MemcpyParams>(op->params);
                if (p.stream) exec_stream = p.stream;
                break;
            }
            case OperationType::MEMSET_ASYNC: {
                auto& p = std::get<MemsetParams>(op->params);
                if (p.stream) exec_stream = p.stream;
                break;
            }
            default:
                break;
        }

        // Green Context 模式下覆盖 stream
        if (current_mode_ == ExecutionMode::ISOLATED && green_ctx_initialized_) {
            exec_stream = worker_stream;
        }

        cudaError_t exec_err = execute_operation(op, exec_stream, worker_cublas_handle);

        // 异步模式：记录错误（不阻塞客户端）
        if (exec_err != cudaSuccess) {
            LOG_ERROR("Worker %d: op %lu (type=%s) failed with error %d: %s",
                      client_idx, op->op_id, op_type_name(op->type),
                      (int)exec_err, cudaGetErrorString(exec_err));
            set_last_error(client_idx, exec_err);
        }

        if (is_kernel_operation(op->type)) {
            std::lock_guard<std::mutex> lock(g_orion_state.mutex);
            g_orion_state.seen[client_idx]++;
        }

        op->mark_completed(exec_err);
    }

    // 处理剩余操作
    LOG_INFO("Worker %d: draining remaining operations", client_idx);
    while (!q->empty()) {
        auto op = q->try_pop();
        if (op) {
            cudaError_t exec_err = execute_operation(op, worker_stream, worker_cublas_handle);
            op->mark_completed(exec_err);
        }
    }
    if (worker_stream) cudaStreamSynchronize(worker_stream);

    LOG_INFO("Worker %d exiting", client_idx);
}

/**
 * @brief 单线程轮询调度器主循环（动态模式切换版）
 *
 * 每轮循环：
 * 1. 模式在初始化时已静态确定（DEFAULT 或全程 ISOLATED），运行时不再切换
 * 2. 轮询所有客户端队列，根据当前模式选择 stream/handle/context 执行
 * 3. 仅在 ISOLATED 模式且 context 实际变化时才调用 cuCtxSetCurrent
 */
void Scheduler::run() {
    tl_is_scheduler_thread = true;

    LOG_INFO("[MODE] Scheduler thread running (GC %s, mode: %s)",
             green_ctx_initialized_ ? "available" : "unavailable",
             current_mode_ == ExecutionMode::DEFAULT ? "DEFAULT" : "ISOLATED");

    unsigned long loop_count = 0;
    unsigned long default_concurrent_count = 0;

    while (running_.load()) {
        bool did_work = false;
        loop_count++;

        // 轮询所有客户端队列（模式已静态固定，无需每轮决策）
        for (int i = 0; i < num_clients_; i++) {
            if (!g_capture_state.client_queues[i]) {
                if (i == 1 && loop_count % 10000 == 0) {
                    LOG_INFO("[SCHEDULER] Client 1 queue is null!");
                }
                continue;
            }

            auto op = g_capture_state.client_queues[i]->peek();
            if (!op) {
                if (i == 1 && loop_count % 10000 == 0) {
                    LOG_INFO("[SCHEDULER] Client 1 queue is empty (loop %lu)", loop_count);
                }
                continue;
            }

            if (i == 1) {
                LOG_INFO("[SCHEDULER] Client 1 has op, type=%d, checking schedule...", (int)op->type);
            }

            // BE 调度判断：DEFAULT 模式基于 SM 冲突，ISOLATED 模式始终允许
            if (i > 0 && !orion_should_schedule(op, i)) {
                if (i == 1) {
                    LOG_INFO("[SCHEDULER] Client 1 op BLOCKED by should_schedule");
                }
                continue;
            }

            LOG_INFO("[SCHEDULER] Dequeuing op from client %d, type=%d", i, (int)op->type);

            g_capture_state.client_queues[i]->try_pop();

            // 选择当前模式对应的 stream 和 cuBLAS handle
            cudaStream_t stream;
            cublasHandle_t cublas_handle = nullptr;

            if (current_mode_ == ExecutionMode::ISOLATED) {
                stream = (i == 0) ? hp_gc_stream_ : be_gc_streams_[0];
                int ctx_idx = (i == 0) ? 0 : 1;
                cublas_handle = cublas_handles_[ctx_idx];

                if (is_kernel_operation(op->type) && ctx_idx != current_ctx_idx_) {
                    LOG_INFO("[SCHEDULER] Switching context from %d to %d", current_ctx_idx_, ctx_idx);
                    cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
                    current_ctx_idx_ = ctx_idx;
                    LOG_INFO("[SCHEDULER] Context switched to %d", ctx_idx);
                }
            } else {
                stream = (i == 0) ? hp_default_stream_ : be_default_streams_[i - 1];
                if (i > 0 && is_kernel_operation(op->type)) {
                    default_concurrent_count++;
                }
            }

            LOG_INFO("[SCHEDULER] Executing op for client %d", i);
            cudaError_t err = execute_operation(op, stream, cublas_handle);
            LOG_INFO("[SCHEDULER] Op executed for client %d, err=%d", i, (int)err);

            if (is_kernel_operation(op->type)) {
                std::lock_guard<std::mutex> lock(g_orion_state.mutex);
                g_orion_state.seen[i]++;
            }

            op->mark_completed(err);
            did_work = true;
        }

        if (!did_work) {
            std::this_thread::yield();
        }
    }

    // 处理剩余操作（使用当前模式的资源）
    LOG_INFO("[MODE] Draining remaining ops in %s mode (ran %lu loops)",
             current_mode_ == ExecutionMode::DEFAULT ? "DEFAULT" : "ISOLATED",
             loop_count);

    for (int i = 0; i < num_clients_; i++) {
        if (!g_capture_state.client_queues[i]) continue;

        cudaStream_t stream;
        cublasHandle_t cublas_handle = nullptr;

        if (current_mode_ == ExecutionMode::ISOLATED) {
            stream = (i == 0) ? hp_gc_stream_ : be_gc_streams_[0];
            int ctx_idx = (i == 0) ? 0 : 1;
            cublas_handle = cublas_handles_[ctx_idx];
        } else {
            stream = (i == 0) ? hp_default_stream_ : be_default_streams_[i - 1];
        }

        while (!g_capture_state.client_queues[i]->empty()) {
            auto op = g_capture_state.client_queues[i]->try_pop();
            if (op) {
                if (current_mode_ == ExecutionMode::ISOLATED && is_kernel_operation(op->type)) {
                    int ctx_idx = (i == 0) ? 0 : 1;
                    if (ctx_idx != current_ctx_idx_) {
                        cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
                        current_ctx_idx_ = ctx_idx;
                    }
                }
                cudaError_t err = execute_operation(op, stream, cublas_handle);
                op->mark_completed(err);
            }
        }
        if (stream) cudaStreamSynchronize(stream);
    }

    LOG_INFO("[MODE] Scheduler thread exiting (final mode: %s, loops=%lu, default_concurrent=%lu)",
             current_mode_ == ExecutionMode::DEFAULT ? "DEFAULT" : "ISOLATED",
             loop_count, default_concurrent_count);
}

// ============================================================================
// 便捷函数
// ============================================================================

bool start_scheduler(int num_clients, const SchedulerConfig& config) {
    if (init_capture_layer(num_clients) != 0) {
        LOG_ERROR("Failed to initialize capture layer");
        return false;
    }

    if (!g_scheduler.init(num_clients, config)) {
        LOG_ERROR("Failed to initialize scheduler");
        return false;
    }

    g_scheduler.start();
    return true;
}

void stop_scheduler() {
    g_scheduler.stop();
    g_scheduler.join();
    g_scheduler.reset();
    shutdown_capture_layer();
}

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_init_scheduler(int num_clients) {
    orion::SchedulerConfig config;
    if (orion::init_capture_layer(num_clients) != 0) {
        LOG_ERROR("Failed to initialize capture layer");
        return -1;
    }
    if (!orion::g_scheduler.init(num_clients, config)) {
        LOG_ERROR("Failed to initialize scheduler");
        return -1;
    }
    return 0;
}

int orion_start_scheduler_thread() {
    orion::g_scheduler.start();
    return 0;
}

int orion_start_scheduler(int num_clients) {
    orion::SchedulerConfig config;
    return orion::start_scheduler(num_clients, config) ? 0 : -1;
}

void orion_stop_scheduler() {
    orion::stop_scheduler();
}

int orion_load_kernel_info(int client_idx, const char* file_path) {
    if (client_idx < 0 || client_idx >= (int)orion::g_orion_state.op_info_vector.size()) {
        LOG_ERROR("Invalid client_idx %d", client_idx);
        return -1;
    }
    return orion::populate_kernel_info(client_idx, std::string(file_path));
}

void orion_set_client_kernels(int client_idx, int num_kernels) {
    if (client_idx < 0 || client_idx >= (int)orion::g_orion_state.num_client_kernels.size()) {
        LOG_ERROR("Invalid client_idx %d", client_idx);
        return;
    }
    std::lock_guard<std::mutex> lock(orion::g_orion_state.mutex);
    orion::g_orion_state.num_client_kernels[client_idx] = num_kernels;
    LOG_INFO("Client %d: num_kernels=%d", client_idx, num_kernels);
}

void orion_set_sm_threshold(int threshold) {
    std::lock_guard<std::mutex> lock(orion::g_orion_state.mutex);
    orion::g_orion_state.sm_threshold = threshold;
    orion::g_scheduler.get_mutable_config().sm_threshold = threshold;
    LOG_INFO("SM threshold set to %d", threshold);
}

int orion_get_sm_threshold() {
    std::lock_guard<std::mutex> lock(orion::g_orion_state.mutex);
    return orion::g_orion_state.sm_threshold;
}

void orion_sync_client_stream(int client_idx) {
    orion::g_scheduler.sync_client_stream(client_idx);
}

void orion_reset_state() {
    orion::g_orion_state.reset();
    LOG_INFO("Orion state reset");
}

int orion_autotune_green_ctx() {
    return orion::g_scheduler.autotune_green_ctx();
}

} // extern "C"
