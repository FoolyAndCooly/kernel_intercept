/**
 * @file gpu_capture.cpp
 * @brief Orion GPU 操作捕获层实现（简化版）
 *
 * 架构：
 * - 客户端线程提交操作到队列，然后忙等完成
 * - 调度器线程轮询队列执行操作
 */

#include "gpu_capture.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

namespace orion {

// ============================================================================
// 全局变量
// ============================================================================

CaptureLayerState g_capture_state;
LogLevel g_log_level = LogLevel::INFO;

// 线程局部变量
static thread_local int tl_client_idx = -1;
thread_local bool tl_is_scheduler_thread = false;
thread_local bool tl_in_scheduler_execution = false;

// ----------------------------------------------------------------------------
// Stream / handle → client_idx 映射（Step B：供库内部线程回查 client）
// 这里先提供存储和查询实现，wrapper 在 Step B 切换到 resolve_client_idx()。
// ----------------------------------------------------------------------------
static std::unordered_map<void*, int> g_stream_to_client;
static std::unordered_map<void*, int> g_handle_to_client;
static std::shared_mutex              g_client_map_mutex;

void register_client_stream(int client_idx, void* stream) {
    if (client_idx < 0 || stream == nullptr) return;
    std::unique_lock lk(g_client_map_mutex);
    g_stream_to_client[stream] = client_idx;
}

void register_client_handle(int client_idx, void* handle) {
    if (client_idx < 0 || handle == nullptr) return;
    std::unique_lock lk(g_client_map_mutex);
    g_handle_to_client[handle] = client_idx;
}

int resolve_client_idx(void* stream, void* handle) {
    if (tl_client_idx >= 0) {
        return tl_client_idx;
    }
    std::shared_lock lk(g_client_map_mutex);
    if (stream) {
        auto it = g_stream_to_client.find(stream);
        if (it != g_stream_to_client.end()) {
            return it->second;
        }
    }
    if (handle) {
        auto it = g_handle_to_client.find(handle);
        if (it != g_handle_to_client.end()) {
            return it->second;
        }
    }
    LOG_DEBUG("[RESOLVE] No client found (stream=%p, handle=%p, tl_client_idx=%d)", stream, handle, tl_client_idx);
    return -1;
}

void clear_client_maps() {
    std::unique_lock lk(g_client_map_mutex);
    g_stream_to_client.clear();
    g_handle_to_client.clear();
}

// ============================================================================
// 日志初始化
// ============================================================================

void init_log_level() {
    const char* level_str = std::getenv("ORION_LOG_LEVEL");
    if (level_str) {
        if (strcmp(level_str, "NONE") == 0 || strcmp(level_str, "0") == 0) {
            g_log_level = LogLevel::NONE;
        } else if (strcmp(level_str, "ERROR") == 0 || strcmp(level_str, "1") == 0) {
            g_log_level = LogLevel::ERROR;
        } else if (strcmp(level_str, "WARN") == 0 || strcmp(level_str, "2") == 0) {
            g_log_level = LogLevel::WARN;
        } else if (strcmp(level_str, "INFO") == 0 || strcmp(level_str, "3") == 0) {
            g_log_level = LogLevel::INFO;
        } else if (strcmp(level_str, "DEBUG") == 0 || strcmp(level_str, "4") == 0) {
            g_log_level = LogLevel::DEBUG;
        } else if (strcmp(level_str, "TRACE") == 0 || strcmp(level_str, "5") == 0) {
            g_log_level = LogLevel::TRACE;
        }
    }
}

// ============================================================================
// 捕获层初始化和关闭
// ============================================================================

int init_capture_layer(int num_clients) {
    if (g_capture_state.initialized.load()) {
        LOG_WARN("Capture layer already initialized");
        return 0;
    }

    if (num_clients <= 0 || num_clients > MAX_CLIENTS) {
        LOG_ERROR("Invalid num_clients: %d (must be 1-%d)", num_clients, MAX_CLIENTS);
        return -1;
    }

    init_log_level();

    // 初始化异步模式（从环境变量读取）
    const char* async_mode_env = std::getenv("ORION_ASYNC_MODE");
    if (async_mode_env) {
        int mode = std::atoi(async_mode_env);
        if (mode >= 0 && mode <= 2) {
            set_async_mode_internal(mode);
            LOG_INFO("Async mode set to %d from ORION_ASYNC_MODE env var", mode);
        } else {
            LOG_WARN("Invalid ORION_ASYNC_MODE value: %s (must be 0-2)", async_mode_env);
        }
    }

    g_capture_state.num_clients = num_clients;

    // 初始化 per-client 操作队列
    g_capture_state.client_queues.resize(num_clients);
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_queues[i] = std::make_unique<ClientQueue>();
    }

    g_capture_state.shutdown.store(false);
    g_capture_state.initialized.store(true);
    g_capture_state.enabled.store(true);

    LOG_INFO("Capture layer initialized with %d clients", num_clients);
    return 0;
}

void shutdown_capture_layer() {
    if (!g_capture_state.initialized.load()) {
        return;
    }

    LOG_INFO("Shutting down capture layer...");

    g_capture_state.shutdown.store(true);
    g_capture_state.enabled.store(false);

    for (int i = 0; i < g_capture_state.num_clients; i++) {
        g_capture_state.client_queues[i]->shutdown();
    }

    g_capture_state.initialized.store(false);
    LOG_INFO("Capture layer shutdown complete");
}

// ============================================================================
// Client 管理
// ============================================================================

int get_current_client_idx() {
    return tl_client_idx;
}

void set_current_client_idx(int idx) {
    if (idx < -1 || idx >= g_capture_state.num_clients) {
        LOG_ERROR("Invalid client index: %d", idx);
        return;
    }
    tl_client_idx = idx;
    LOG_DEBUG("Thread set to client %d", idx);
}

bool is_managed_thread() {
    return tl_client_idx >= 0 && g_capture_state.enabled.load();
}

bool is_capture_enabled() {
    return g_capture_state.initialized.load() && g_capture_state.enabled.load();
}

void set_capture_enabled(bool enabled) {
    g_capture_state.enabled.store(enabled);
    LOG_INFO("Capture %s", enabled ? "enabled" : "disabled");
}

// ============================================================================
// 操作提交
// ============================================================================

OperationPtr create_operation(int client_idx, OperationType type) {
    if (client_idx < 0 || client_idx >= g_capture_state.num_clients) {
        LOG_ERROR("Invalid client index for create: %d", client_idx);
        return nullptr;
    }

    auto op = std::make_shared<OperationRecord>();
    op->type = type;
    op->client_idx = client_idx;
    op->op_id = g_capture_state.next_op_id.fetch_add(1);

    LOG_DEBUG("Created op %lu type %s for client %d", op->op_id, op_type_name(type), client_idx);
    return op;
}

void enqueue_operation(OperationPtr op) {
    if (!op) return;

    if (op->client_idx == 1) {
        LOG_INFO("[ENQUEUE] Client 1 enqueuing op %lu type %s", op->op_id, op_type_name(op->type));
    }

    g_capture_state.client_queues[op->client_idx]->push(op);

    if (op->client_idx == 1) {
        LOG_INFO("[ENQUEUE] Client 1 op %lu pushed to queue", op->op_id);
    }
}

void wait_operation(OperationPtr op) {
    if (!op) return;
    op->wait_completion();  // 忙等
}

// 兼容旧接口
OperationPtr submit_operation(int client_idx, OperationType type) {
    auto op = create_operation(client_idx, type);
    if (op) {
        enqueue_operation(op);
    }
    return op;
}

// 异步模式控制（默认 Level 1: Worker 异步）
static std::atomic<int> g_async_mode{1};

// 异步模式错误记录（每个 client 一个）
constexpr int MAX_CLIENTS = 16;
static std::atomic<cudaError_t> g_last_error[MAX_CLIENTS];

void set_async_mode_internal(int mode) {
    g_async_mode.store(mode);
    LOG_INFO("Async mode set to %d (0=sync, 1=worker_async, 2=passthrough)", mode);
}

int get_async_mode_internal() {
    return g_async_mode.load();
}

void set_last_error(int client_idx, cudaError_t error) {
    if (client_idx >= 0 && client_idx < MAX_CLIENTS) {
        // 只记录第一个错误（后续错误被忽略）
        cudaError_t expected = cudaSuccess;
        g_last_error[client_idx].compare_exchange_strong(expected, error);
    }
}

cudaError_t get_and_clear_last_error(int client_idx) {
    if (client_idx >= 0 && client_idx < MAX_CLIENTS) {
        return g_last_error[client_idx].exchange(cudaSuccess);
    }
    return cudaSuccess;
}

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_init(int num_clients) {
    return orion::init_capture_layer(num_clients);
}

void orion_shutdown() {
    orion::shutdown_capture_layer();
}

void orion_set_client_idx(int idx) {
    orion::set_current_client_idx(idx);
}

int orion_get_client_idx() {
    return orion::get_current_client_idx();
}

void orion_set_enabled(int enabled) {
    orion::set_capture_enabled(enabled != 0);
}

// 别名：方案文档里使用 orion_set_capture，保留 orion_set_enabled 作为旧名
int orion_set_capture(int enabled) {
    orion::set_capture_enabled(enabled != 0);
    return 0;
}

// Step B 预备：供 Python / 调度器显式登记 stream/handle 与 client 的映射
void orion_register_client_stream(int client_idx, void* stream) {
    orion::register_client_stream(client_idx, stream);
}

void orion_register_client_handle(int client_idx, void* handle) {
    orion::register_client_handle(client_idx, handle);
}

int orion_resolve_client_idx(void* stream, void* handle) {
    return orion::resolve_client_idx(stream, handle);
}

int orion_is_enabled() {
    return orion::is_capture_enabled() ? 1 : 0;
}

} // extern "C"
