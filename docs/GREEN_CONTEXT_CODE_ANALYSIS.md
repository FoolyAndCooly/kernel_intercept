# Orion 调度器 Green Context 实现代码分析

**文档版本**: 1.0
**分析日期**: 2026-03-04
**项目路径**: `/home/liuyc/kernel_intercept`

---

## 目录

1. [概述](#1-概述)
2. [项目文件结构](#2-项目文件结构)
3. [核心数据结构](#3-核心数据结构)
4. [Green Context 初始化流程](#4-green-context-初始化流程)
5. [调度器主循环中的 Context 切换](#5-调度器主循环中的-context-切换)
6. [cuBLAS Handle 的 Context 亲和性处理](#6-cublas-handle-的-context-亲和性处理)
7. [资源销毁与清理](#7-资源销毁与清理)
8. [Fallback 机制](#8-fallback-机制)
9. [环境变量配置接口](#9-环境变量配置接口)
10. [测试与验证体系](#10-测试与验证体系)
11. [关键设计决策分析](#11-关键设计决策分析)
12. [性能测试结果](#12-性能测试结果)
13. [完整调用链路图](#13-完整调用链路图)

---

## 1. 概述

### 1.1 问题背景

Orion 调度器通过 `LD_PRELOAD` 拦截 CUDA/cuBLAS/cuDNN API，将多个客户端（1 HP + N BE）的 GPU 操作重定向到调度器管理的 CUDA Stream 上执行。在原始方案中，HP 和 BE 的 kernel 共享所有 114 个 SM（H100 GPU），仅依靠 Stream Priority 做软件层面的优先级控制。这导致：

- 大型 GEMM kernel（需 64+ SM）在 HP 和 BE 间产生严重的硬件层面 SM 争抢
- 中型 GEMM 并发执行时耗时膨胀超过 100%
- Stream Priority 只影响同一 SM 上的 warp 调度优先级，无法阻止两个 stream 的 kernel 同时占据同一个 SM

### 1.2 解决方案

集成 CUDA 12.4+ 的 **Green Context** API，实现硬件级 SM 分区隔离。Green Context 是 CUDA Driver API 特性（需 Hopper sm_90+ 架构），允许将 GPU 的 SM 资源按数量切分为多个不重叠的分区，每个分区绑定一个独立的 `CUcontext`。在某个 Green Context 中启动的 kernel 只能使用该分区内的 SM，由硬件强制保证。

### 1.3 架构对比

```
传统模式 (Stream Priority):             Green Context 模式 (SM 隔离):
┌──────────────────────────────┐       ┌──────────────────────────────┐
│  GPU (114 SMs)               │       │  GPU (114 SMs)               │
│  ┌────────────────────────┐  │       │  ┌──────────┐ ┌──────────┐  │
│  │ HP GEMM + BE GEMM     │  │       │  │ HP GEMM  │ │ BE GEMM  │  │
│  │ 共享全部 SM，互相争抢   │  │       │  │ 56 SMs   │ │ 56 SMs   │  │
│  └────────────────────────┘  │       │  │ 硬件独占  │ │ 硬件独占  │  │
│                              │       │  └──────────┘ └──────────┘  │
└──────────────────────────────┘       └──────────────────────────────┘
```

---

## 2. 项目文件结构

与 Green Context 相关的所有文件及其职责：

### 2.1 核心源码

| 文件 | 职责 |
|------|------|
| `include/scheduler.h` | 定义 `GreenCtxConfig` 结构体和 `Scheduler` 类中的 Green Context 成员变量 |
| `src/scheduler.cpp` | Green Context 的完整生命周期实现：初始化、Context 切换、资源销毁、Fallback 逻辑 |
| `src/cublas_intercept.cpp` | cuBLAS 拦截层中对 Green Context 提供的 cuBLAS handle 的使用 |
| `include/gpu_capture.h` | 操作记录结构体定义（`CublasGemmParams` 等），不直接涉及 Green Context 但提供数据流基础 |

### 2.2 测试代码

| 文件 | 职责 |
|------|------|
| `tests/test_green_context_api.cu` | 独立测试 Green Context API 可用性（SM 分割、Green Context 创建） |
| `tests/test_green_context_simple.cu` | 简化的 SM 分区测试（验证 `cuDevSmResourceSplitByCount` 的分割逻辑） |
| `test_scheduler_green_ctx.py` | Python 端到端测试：加载调度器库、启动 PyTorch 推理、验证 Green Context 是否生效 |
| `test_green_context_comparison.sh` | 自动化对比测试脚本：Baseline vs Green Context 全流程 |

### 2.3 辅助脚本

| 文件 | 职责 |
|------|------|
| `run.sh` | 项目启动脚本，通过 `LD_PRELOAD` 加载调度器库执行测试 |
| `generate_comparison_report.py` | 从日志中提取 Baseline 和 Green Context 结果，生成对比报告 |
| `generate_updated_report.py` | 多配置（48:48, 56:56）对比报告生成器 |

### 2.4 构建系统

| 文件 | 职责 |
|------|------|
| `Makefile` | 编译 `libgpu_scheduler.so`，链接 `-lcuda`（CUDA Driver API）以支持 Green Context |

### 2.5 文档与报告

| 文件 | 职责 |
|------|------|
| `docs/green_context_design.md` | 设计文档：问题分析、方案设计、验证计划 |
| `GREEN_CONTEXT_TEST_SUMMARY.md` | 测试总结：功能验证结论 |
| `green_context_results/GREEN_CONTEXT_PERFORMANCE_REPORT.md` | 性能报告 v1（48:48 配置） |
| `green_context_results/GREEN_CONTEXT_PERFORMANCE_REPORT_UPDATED.md` | 性能报告 v2（48:48 + 56:56 对比） |

---

## 3. 核心数据结构

### 3.1 GreenCtxConfig — SM 分区配置

**文件**: `include/scheduler.h` (第 104-108 行)

```cpp
struct GreenCtxConfig {
    unsigned int hp_sm_count = 64;  // HP 分区 SM 数量
    unsigned int be_sm_count = 48;  // BE 分区 SM 数量
    bool enabled = false;           // 是否启用 Green Context
};
```

该结构体封装了 SM 分区的用户配置，在调度器启动时从环境变量 `ORION_HP_SMS` 和 `ORION_BE_SMS` 读取。`enabled` 标志仅在两个环境变量均非零时设为 `true`。

### 3.2 Scheduler 类中的 Green Context 成员

**文件**: `include/scheduler.h` (第 147-156 行)

```cpp
// Green Context 资源
CUdevice cu_device_ = 0;                          // CUDA 设备句柄
CUdevResource full_sm_resource_{};                 // 全部 SM 资源描述符
CUdevResource sm_groups_[2]{};                     // 分割后的两组 SM 资源
CUdevResourceDesc sm_descs_[2] = {nullptr, nullptr}; // 资源描述符
CUgreenCtx green_ctxs_[2] = {nullptr, nullptr};   // Green Context [0]=HP, [1]=BE
CUcontext cuda_ctxs_[2] = {nullptr, nullptr};      // 从 Green Context 派生的 CUcontext
cublasHandle_t cublas_handles_[2] = {nullptr, nullptr}; // 每个 context 绑定一个 cuBLAS handle
bool green_ctx_initialized_ = false;               // 初始化完成标志
GreenCtxConfig gc_config_;                          // 保存的配置
```

设计要点：

- 使用固定大小数组 `[2]`，索引 0 为 HP、索引 1 为 BE，简洁高效
- `green_ctx_initialized_` 作为全局开关，在调度器主循环、Stream 销毁等多处使用
- 每个 Green Context 绑定独立的 `cublasHandle_t`，避免跨 context 使用 handle 导致的错误

### 3.3 操作参数中的 cuBLAS Handle 传递

**文件**: `include/gpu_capture.h` (第 140-167 行)

`CublasGemmParams` 和 `CublasLtMatmulParams` 结构体中保存了拦截时的原始 `handle`。调度器执行时会优先使用 Green Context 绑定的 handle 替换之：

```cpp
struct CublasGemmParams {
    void* handle;           // 原始 cuBLAS handle（拦截时保存）
    // ... 矩阵参数 ...
    float alpha_value;      // 异步模式存储的标量值
    float beta_value;
    bool use_stored_scalars;
};
```

---

## 4. Green Context 初始化流程

### 4.1 入口：`create_streams()`

**文件**: `src/scheduler.cpp` (第 358-416 行)

`create_streams()` 是初始化的入口。它首先尝试读取 Green Context 配置；若配置启用且初始化成功，则使用 Green Context 方案；否则 fallback 到传统的 Stream Priority 方案。

```cpp
bool Scheduler::create_streams() {
    auto gc_config = read_green_ctx_config();
    if (gc_config.enabled) {
        bool ok = init_green_contexts(gc_config);
        if (ok) {
            LOG_INFO("Green Context SM isolation enabled");
            return true;
        }
        LOG_WARN("Green Context init failed, falling back to stream priority");
    }
    // Fallback: Stream Priority 方案 ...
}
```

### 4.2 环境变量读取：`read_green_ctx_config()`

**文件**: `src/scheduler.cpp` (第 225-239 行)

```cpp
static Scheduler::GreenCtxConfig read_green_ctx_config() {
    Scheduler::GreenCtxConfig gc;
    const char* hp = std::getenv("ORION_HP_SMS");
    const char* be = std::getenv("ORION_BE_SMS");
    if (hp && be) {
        gc.hp_sm_count = std::atoi(hp);
        gc.be_sm_count = std::atoi(be);
        gc.enabled = (gc.hp_sm_count > 0 && gc.be_sm_count > 0);
    }
    return gc;
}
```

只有当 `ORION_HP_SMS` 和 `ORION_BE_SMS` 两个环境变量同时设置且值大于 0 时，Green Context 才会启用。

### 4.3 核心初始化：`init_green_contexts()`

**文件**: `src/scheduler.cpp` (第 241-331 行)

这是 Green Context 的核心初始化函数，共分 6 个步骤：

#### Step 1: 初始化 CUDA Driver API

```cpp
CHECK_CU(cuInit(0));
CHECK_CU(cuDeviceGet(&cu_device_, config_.device_id));
```

调用 `cuInit(0)` 初始化 Driver API，然后获取设备句柄。

#### Step 2: 获取全部 SM 资源

```cpp
CHECK_CU(cuDeviceGetDevResource(
    cu_device_, &full_sm_resource_, CU_DEV_RESOURCE_TYPE_SM));
```

调用 `cuDeviceGetDevResource` 获取设备的全部 SM 资源描述符。这是 Green Context API 的基础，后续的 SM 分割都基于此资源。

#### Step 3: SM 数量对齐

```cpp
const unsigned unit = 8;
unsigned int hp_sms = ((gc_config.hp_sm_count + unit - 1) / unit) * unit;
unsigned int be_sms = ((gc_config.be_sm_count + unit - 1) / unit) * unit;

if (hp_sms + be_sms > (unsigned)config_.num_sms) {
    LOG_ERROR("SM partition exceeds total: HP(%u) + BE(%u) > %d",
              hp_sms, be_sms, config_.num_sms);
    return false;
}
```

Hopper 架构的最小 SM 分配单位是 8（1 个 GPC），因此需要向上对齐到 8 的倍数。同时验证分区总数不超过 GPU 总 SM 数。

#### Step 4: 分割 SM 资源

```cpp
unsigned int sm_per_group = (hp_sms < be_sms) ? hp_sms : be_sms;
CUdevResource remaining{};
unsigned int nbGroups = 2;

CHECK_CU(cuDevSmResourceSplitByCount(
    sm_groups_,        // 输出：sm_groups_[0] 和 sm_groups_[1]
    &nbGroups,         // 输入/输出：请求 2 组
    &full_sm_resource_,
    &remaining,
    0,                 // useFlags
    sm_per_group));    // 每组的 SM 数

if (nbGroups < 2) {
    LOG_ERROR("Failed to split SM resource into 2 groups (got %u)", nbGroups);
    return false;
}
```

`cuDevSmResourceSplitByCount` 将全部 SM 一次性分为 2 组，每组 `sm_per_group` 个 SM。该 API 要求每组大小相同，所以取 `min(hp_sms, be_sms)` 作为每组大小。

**设计注意**: 这意味着实际运行时 HP 和 BE 获得相同数量的 SM，即使配置的 `hp_sm_count != be_sm_count`。配置中的较大值会被截断。

#### Step 5: 创建 Green Context

```cpp
for (int i = 0; i < 2; i++) {
    CHECK_CU(cuDevResourceGenerateDesc(&sm_descs_[i], &sm_groups_[i], 1));
    CHECK_CU(cuGreenCtxCreate(&green_ctxs_[i], sm_descs_[i],
                               cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
    CHECK_CU(cuCtxFromGreenCtx(&cuda_ctxs_[i], green_ctxs_[i]));
}
```

对每组 SM 资源：
1. `cuDevResourceGenerateDesc` — 从 SM 资源生成资源描述符
2. `cuGreenCtxCreate` — 基于描述符创建 Green Context，指定 `CU_GREEN_CTX_DEFAULT_STREAM` 标志
3. `cuCtxFromGreenCtx` — 从 Green Context 派生出标准 `CUcontext`，用于后续的 `cuCtxSetCurrent` 切换

#### Step 6: 在各 Context 中创建 Stream 和 cuBLAS Handle

**HP Context (索引 0)**:

```cpp
CHECK_CU(cuCtxSetCurrent(cuda_ctxs_[0]));
CHECK_CUDA(cudaSetDevice(config_.device_id));
CUstream cu_hp_stream;
CHECK_CU(cuGreenCtxStreamCreate(&cu_hp_stream, green_ctxs_[0],
                                 CU_STREAM_NON_BLOCKING, 0));
hp_stream_ = reinterpret_cast<cudaStream_t>(cu_hp_stream);

cublasStatus_t cublas_err = cublasCreate(&cublas_handles_[0]);
```

**BE Context (索引 1)**:

```cpp
CHECK_CU(cuCtxSetCurrent(cuda_ctxs_[1]));
CHECK_CUDA(cudaSetDevice(config_.device_id));
CUstream cu_be_stream;
CHECK_CU(cuGreenCtxStreamCreate(&cu_be_stream, green_ctxs_[1],
                                 CU_STREAM_NON_BLOCKING, 0));
be_streams_.resize(1);
be_streams_[0] = reinterpret_cast<cudaStream_t>(cu_be_stream);

cublas_err = cublasCreate(&cublas_handles_[1]);
```

关键细节：
- 使用 `cuGreenCtxStreamCreate` 而非 `cudaStreamCreate`，确保 stream 绑定到 Green Context
- Stream 通过 `reinterpret_cast` 转换为 `cudaStream_t`，与 Runtime API 兼容
- 在对应 context 激活时创建 `cublasHandle_t`，保证 handle 绑定到正确的 context
- BE 只创建一个 stream（`be_streams_[0]`），所有 BE 客户端共享同一个 Green Context

---

## 5. 调度器主循环中的 Context 切换

### 5.1 主循环：`Scheduler::run()`

**文件**: `src/scheduler.cpp` (第 561-646 行)

调度器使用单线程轮询模型。在执行 kernel 操作前，根据客户端类型切换到对应的 Green Context：

```cpp
void Scheduler::run() {
    tl_is_scheduler_thread = true;

    while (running_.load()) {
        bool did_work = false;

        for (int i = 0; i < num_clients_; i++) {
            if (!g_capture_state.client_queues[i]) continue;

            auto op = g_capture_state.client_queues[i]->peek();
            if (!op) continue;

            cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i - 1];
            bool should_execute = (i == 0) || orion_should_schedule(op, i);

            if (should_execute) {
                g_capture_state.client_queues[i]->try_pop();

                // ★ 关键：执行前切换到对应的 Green Context
                int ctx_idx = (i == 0) ? 0 : 1;
                if (green_ctx_initialized_ && is_kernel_operation(op->type)) {
                    cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
                    cudaSetDevice(config_.device_id);
                }

                // 传递对应的 cuBLAS handle
                cublasHandle_t cublas_handle =
                    green_ctx_initialized_ ? cublas_handles_[ctx_idx] : nullptr;
                cudaError_t err = execute_operation(op, stream, cublas_handle);

                // 更新调度计数
                if (is_kernel_operation(op->type)) {
                    std::lock_guard<std::mutex> lock(g_orion_state.mutex);
                    g_orion_state.seen[i]++;
                }

                op->mark_completed(err);
                did_work = true;
            }
        }

        if (!did_work) {
            std::this_thread::yield();
        }
    }
    // ... 退出时处理剩余操作 ...
}
```

### 5.2 Context 切换逻辑分析

```cpp
int ctx_idx = (i == 0) ? 0 : 1;
if (green_ctx_initialized_ && is_kernel_operation(op->type)) {
    cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
    cudaSetDevice(config_.device_id);
}
```

- **客户端映射**: 客户端 0 (HP) → `ctx_idx=0`；客户端 1+ (BE) → `ctx_idx=1`。所有 BE 客户端共享同一个 Green Context
- **条件切换**: 仅在 Green Context 已初始化且操作是 kernel 类型时才切换。内存操作（`malloc`/`memcpy` 等）不需要 context 切换
- **性能开销**: `cuCtxSetCurrent()` 是线程局部操作，仅修改调用线程的 context 指针，不涉及 GPU 端同步，实测开销约 5 μs，远小于 kernel 执行时间

### 5.3 退出时的剩余操作处理

```cpp
for (int i = 0; i < num_clients_; i++) {
    if (!g_capture_state.client_queues[i]) continue;

    cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i - 1];
    int ctx_idx = (i == 0) ? 0 : 1;
    cublasHandle_t cublas_handle =
        green_ctx_initialized_ ? cublas_handles_[ctx_idx] : nullptr;

    while (!g_capture_state.client_queues[i]->empty()) {
        auto op = g_capture_state.client_queues[i]->try_pop();
        if (op) {
            if (green_ctx_initialized_ && is_kernel_operation(op->type)) {
                cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
                cudaSetDevice(config_.device_id);
            }
            cudaError_t err = execute_operation(op, stream, cublas_handle);
            op->mark_completed(err);
        }
    }
    cudaStreamSynchronize(stream);
}
```

在调度器停止时，同样遵循 Green Context 切换逻辑处理队列中的剩余操作。

### 5.4 `is_kernel_operation()` 判断

**文件**: `src/scheduler.cpp` (第 60-76 行)

```cpp
static bool is_kernel_operation(OperationType type) {
    switch (type) {
        case OperationType::KERNEL_LAUNCH:
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
```

只有会实际使用 SM 的计算类操作才需要 context 切换；内存操作、同步操作等不需要。

---

## 6. cuBLAS Handle 的 Context 亲和性处理

### 6.1 问题

cuBLAS 的 `cublasHandle_t` 在创建时绑定到当前活跃的 CUDA context。如果在一个 context 中创建的 handle 被用于另一个 context 中执行操作，将导致未定义行为或错误。

### 6.2 解决方案

在 `init_green_contexts()` 的 Step 6 中，分别在 HP 和 BE 的 context 中创建独立的 cuBLAS handle：

```cpp
// HP context
cuCtxSetCurrent(cuda_ctxs_[0]);
cublasCreate(&cublas_handles_[0]);

// BE context
cuCtxSetCurrent(cuda_ctxs_[1]);
cublasCreate(&cublas_handles_[1]);
```

### 6.3 Handle 传递链路

**调度器主循环** (`scheduler.cpp`):

```cpp
cublasHandle_t cublas_handle =
    green_ctx_initialized_ ? cublas_handles_[ctx_idx] : nullptr;
cudaError_t err = execute_operation(op, stream, cublas_handle);
```

**`execute_operation()`** (`scheduler.cpp`):

```cpp
cudaError_t Scheduler::execute_operation(OperationPtr op, cudaStream_t stream,
                                          cublasHandle_t cublas_handle) {
    // ...
    case OperationType::CUBLAS_SGEMM:
    case OperationType::CUBLAS_SGEMM_BATCHED:
    case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
        return execute_cublas_operation(op, stream, (void*)cublas_handle)
            == 0 ? cudaSuccess : cudaErrorUnknown;
    // ...
}
```

**`execute_cublas_operation()`** (`cublas_intercept.cpp`):

```cpp
cublasStatus_t execute_cublas_operation(OperationPtr op,
    cudaStream_t scheduler_stream, cublasHandle_t provided_handle) {
    // 优先使用 Green Context 提供的 handle
    cublasHandle_t handle = provided_handle;

    if (!handle) {
        // Fallback: 使用线程本地的 cuBLAS handle
        handle = get_thread_local_cublas_handle(scheduler_stream);
    }

    if (!handle) {
        // 最后回退到原始 handle
        // ...
    } else if (scheduler_stream && g_cublas_funcs.cublasSetStream_v2) {
        // 为提供的 handle 设置 stream
        g_cublas_funcs.cublasSetStream_v2(handle, scheduler_stream);
    }
    // ... 执行实际 cuBLAS 操作 ...
}
```

handle 选择优先级：**Green Context handle > 线程本地 handle > 原始 handle**。

---

## 7. 资源销毁与清理

### 7.1 `destroy_green_contexts()`

**文件**: `src/scheduler.cpp` (第 333-356 行)

```cpp
void Scheduler::destroy_green_contexts() {
    if (!green_ctx_initialized_) return;

    // 1. 销毁 cuBLAS handles
    for (int i = 0; i < 2; i++) {
        if (cublas_handles_[i]) {
            cublasDestroy(cublas_handles_[i]);
            cublas_handles_[i] = nullptr;
        }
    }

    // 2. 销毁 Green Contexts（stream 随 context 一起销毁）
    for (int i = 0; i < 2; i++) {
        if (green_ctxs_[i]) {
            cuGreenCtxDestroy(green_ctxs_[i]);
            green_ctxs_[i] = nullptr;
        }
        sm_descs_[i] = nullptr;
    }

    green_ctx_initialized_ = false;
}
```

销毁顺序：cuBLAS handle → Green Context。Stream 由 Green Context 管理，随 context 销毁一起释放。

### 7.2 `destroy_streams()` 的分支处理

```cpp
void Scheduler::destroy_streams() {
    if (green_ctx_initialized_) {
        destroy_green_contexts();
        return;
    }
    // Fallback: 清理传统 streams
    if (hp_stream_) { cudaStreamDestroy(hp_stream_); hp_stream_ = nullptr; }
    for (auto& stream : be_streams_) {
        if (stream) { cudaStreamDestroy(stream); }
    }
    be_streams_.clear();
}
```

根据是否使用了 Green Context，走不同的清理路径。

---

## 8. Fallback 机制

### 8.1 设计原则

Green Context 需要 CUDA 12.4+、Hopper (sm_90+) GPU、兼容驱动。在不满足条件时，系统自动 fallback 到传统的 Stream Priority 方案，保证向后兼容。

### 8.2 Fallback 流程

```
create_streams()
    │
    ├── read_green_ctx_config()
    │       │
    │       ├── 环境变量未设置 → gc_config.enabled = false → 跳过 Green Context
    │       └── 环境变量已设置 → gc_config.enabled = true
    │
    ├── gc_config.enabled == true
    │       │
    │       ├── init_green_contexts() 成功 → 使用 Green Context → return true
    │       └── init_green_contexts() 失败 → LOG_WARN → 继续 fallback
    │
    └── Fallback: Stream Priority
            │
            ├── cudaDeviceGetStreamPriorityRange()
            ├── cudaStreamCreateWithPriority(hp_stream_, highest)
            └── cudaStreamCreateWithPriority(be_streams_[i], lower)
```

### 8.3 Fallback 的 Stream Priority 方案

```cpp
int lowest_priority, highest_priority;
cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);

// HP: 最高优先级
cudaStreamCreateWithPriority(&hp_stream_, cudaStreamNonBlocking, highest_priority);

// BE: 优先级递减
for (int i = 0; i < num_be; i++) {
    int be_priority = highest_priority + 1 +
        (i * (priority_range - 1)) / std::max(1, num_be - 1);
    be_priority = std::min(be_priority, lowest_priority);
    cudaStreamCreateWithPriority(&be_streams_[i], cudaStreamNonBlocking, be_priority);
}
```

Fallback 方案为每个 BE 客户端创建独立 stream，按优先级排列。HP stream 获得最高优先级。

---

## 9. 环境变量配置接口

### 9.1 配置说明

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `ORION_HP_SMS` | int | 未设置 | HP 分区请求的 SM 数量 |
| `ORION_BE_SMS` | int | 未设置 | BE 分区请求的 SM 数量 |

### 9.2 SM 对齐规则

用户配置的 SM 数会自动向上对齐到 8 的倍数（Hopper 最小分配单位）：

| 用户配置 | 对齐后 |
|----------|--------|
| 60 | 64 |
| 48 | 48 |
| 50 | 56 |
| 64 | 64 |

### 9.3 实际分区行为

由于 `cuDevSmResourceSplitByCount` 要求每组大小相同，实际每组获得的 SM 数为 `min(对齐后HP, 对齐后BE)`：

| 配置 | HP 对齐 | BE 对齐 | 实际每组 | 总使用 | 剩余 |
|------|---------|---------|---------|--------|------|
| `HP=64, BE=48` | 64 | 48 | 48 | 96 | 18 |
| `HP=56, BE=56` | 56 | 56 | 56 | 112 | 2 |
| `HP=72, BE=40` | 72 | 40 | 40 | 80 | 34 |

### 9.4 使用示例

```bash
# 启用 Green Context (56:56 配置)
ORION_HP_SMS=56 ORION_BE_SMS=56 \
  LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py

# 不设置环境变量 → 自动 fallback 到 Stream Priority
LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py
```

---

## 10. 测试与验证体系

### 10.1 API 可用性测试

**文件**: `tests/test_green_context_api.cu`

独立的 CUDA 程序，验证当前 GPU/Driver 是否支持 Green Context API：

1. `cuInit(0)` — 初始化 Driver API
2. `cuDeviceGetDevResource()` — 获取 SM 资源
3. `cuDevSmResourceSplitByCount()` — 分割 SM（HP 64 + BE 48）
4. `cuDevResourceGenerateDesc()` + `cuGreenCtxCreate()` — 创建 Green Context
5. `cuGreenCtxDestroy()` — 清理

编译与运行：
```bash
nvcc -arch=sm_90 -o test_gc_api tests/test_green_context_api.cu -lcuda
./test_gc_api
```

### 10.2 SM 分区验证测试

**文件**: `tests/test_green_context_simple.cu`

更简化的测试，专注验证 `cuDevSmResourceSplitByCount` 的分割逻辑：先从全部资源分割出 HP 组（56 SMs），然后从剩余资源分割 BE 组（56 或 48 SMs）。

### 10.3 端到端功能测试

**文件**: `test_scheduler_green_ctx.py`

Python 脚本，通过 `ctypes` 加载 `libgpu_scheduler.so`，调用 C 接口启动调度器，然后运行 PyTorch 推理验证功能正确性：

```python
lib = ctypes.CDLL("build/libgpu_scheduler.so")
lib.orion_start_scheduler(2)  # 1 HP + 1 BE

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 2048),
    torch.nn.ReLU(),
    torch.nn.Linear(2048, 1024),
).to("cuda:0")

# 推理测试...
lib.orion_stop_scheduler()
```

### 10.4 性能对比测试

**文件**: `test_green_context_comparison.sh`

自动化脚本，依次运行 Baseline 和 Green Context 测试，提取结果并生成对比报告：

1. Baseline 测试：不设置 `ORION_HP_SMS`/`ORION_BE_SMS`
2. Green Context 测试：设置 `ORION_HP_SMS=64 ORION_BE_SMS=48`
3. 调用 Python 脚本计算性能改善百分比
4. 生成 `comparison_report_{timestamp}.txt`

### 10.5 报告生成工具

- **`generate_comparison_report.py`**: 两配置对比（Baseline vs Green Context）
- **`generate_updated_report.py`**: 三配置对比（Baseline vs 48:48 vs 56:56）

两者都通过正则表达式从日志文件中提取 HP/BE/Total 时间，计算改善百分比和并行效率。

---

## 11. 关键设计决策分析

### 11.1 为什么使用单线程 Context 切换而非多线程

调度器采用单线程轮询模型，在执行不同客户端的 kernel 前通过 `cuCtxSetCurrent` 切换 context。这种设计的优势：

- **无竞争**: 单线程避免了多个线程同时切换 context 的同步问题
- **低开销**: `cuCtxSetCurrent` 是线程局部操作，仅修改指针，不涉及 GPU 同步
- **简单可靠**: 无需引入额外的线程同步机制

### 11.2 为什么所有 BE 共享一个 Green Context

```
ctx_idx = (i == 0) ? 0 : 1;
```

所有 BE 客户端映射到同一个 Green Context（索引 1），原因：
- SM 分区数量有限（Hopper 最多 8 个分区），不适合为每个 BE 分配独立分区
- BE 之间的隔离需求较低，主要隔离目标是 HP vs BE
- 简化资源管理和 context 切换逻辑

### 11.3 为什么仅对 kernel 操作切换 Context

```cpp
if (green_ctx_initialized_ && is_kernel_operation(op->type)) {
    cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
}
```

内存操作（`cudaMalloc`、`cudaMemcpy` 等）不受 SM 分区影响，它们使用的是 GPU 内存控制器而非 SM。因此不需要 context 切换，减少不必要的开销。

### 11.4 为什么使用 `cuDevSmResourceSplitByCount` 而非逐步分割

实际实现中使用一次性分割为两组（`nbGroups = 2`），而非先分 HP 再从剩余中分 BE：

```cpp
unsigned int nbGroups = 2;
CHECK_CU(cuDevSmResourceSplitByCount(
    sm_groups_, &nbGroups, &full_sm_resource_, &remaining, 0, sm_per_group));
```

这种方式的限制是两组必须等大。`test_green_context_api.cu` 中的测试代码展示了另一种两步分割方案，可以实现不等大分区。

### 11.5 Makefile 中的 `-lcuda` 链接

```makefile
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lnvToolsExt
```

Green Context API 属于 CUDA Driver API（`libcuda.so`），需要显式链接 `-lcuda`。原始项目仅依赖 Runtime API (`-lcudart`)。

---

## 12. 性能测试结果

### 12.1 测试环境

- GPU: H100 PCIe (114 SMs, Hopper sm_90)
- CUDA Toolkit: 12.8
- 模型: CharGPT (93.4M 参数)
- 场景: 1 HP + 1 BE 并发推理, 5 迭代

### 12.2 三配置对比

| 配置 | HP 时间 (ms) | BE 时间 (ms) | 总时间 (ms) | 并行效率 |
|------|-------------|-------------|------------|---------|
| Baseline (Stream Priority) | 445.72 | 745.93 | 746.14 | 59.7% |
| Green Context 48:48 | 835.73 | 835.86 | 836.08 | 100.0% |
| Green Context 56:56 | 763.08 | 856.92 | 856.92 | 89.0% |

### 12.3 关键发现

1. **SM 隔离成功**: 48:48 配置下 HP 和 BE 执行时间几乎完全相同（差异 < 0.02%），证明硬件级隔离生效
2. **资源竞争消除**: HP/BE 时间差从 Baseline 的 40% 降至 48:48 的 ~0%，56:56 的 12%
3. **并行效率提升**: 从 59.7% 提升到 89-100%
4. **性能权衡**: HP 延迟增加（SM 减少的预期代价），但换取了隔离性和可预测性
5. **56:56 优于 48:48**: 更多 SM 直接提升了性能，HP 提升 8.7%，SM 利用率 98%

### 12.4 SM 利用效率

| 配置 | 理论性能比 (SM/114) | 实际性能比 | 利用效率 |
|------|---------------------|-----------|---------|
| 48:48 | 42% | 53.3% | 高于理论值 |
| 56:56 | 49% | 58.4% | 高于理论值 |

实际性能优于理论预测，说明 Green Context 下的 SM 独占使得 L2 Cache、内存带宽等共享资源的竞争也减少。

---

## 13. 完整调用链路图

### 13.1 初始化链路

```
main() / orion_start_scheduler()
  └── start_scheduler()
        ├── init_capture_layer()
        └── Scheduler::init()
              ├── cudaGetDeviceProperties() → 获取 SM 数量
              ├── create_streams()
              │     ├── read_green_ctx_config()  → 读 ORION_HP_SMS / ORION_BE_SMS
              │     ├── [启用] init_green_contexts()
              │     │       ├── cuInit() + cuDeviceGet()
              │     │       ├── cuDeviceGetDevResource()      → 获取全部 SM
              │     │       ├── SM 对齐到 8 的倍数
              │     │       ├── cuDevSmResourceSplitByCount()  → 分割为 2 组
              │     │       ├── cuDevResourceGenerateDesc()    → 生成资源描述符 ×2
              │     │       ├── cuGreenCtxCreate()             → 创建 Green Context ×2
              │     │       ├── cuCtxFromGreenCtx()            → 派生 CUcontext ×2
              │     │       ├── cuGreenCtxStreamCreate()       → 创建绑定 Stream ×2
              │     │       └── cublasCreate()                 → 创建 cuBLAS Handle ×2
              │     └── [禁用/失败] Fallback
              │             └── cudaStreamCreateWithPriority() → 传统 Stream
              └── Scheduler::start() → 启动调度线程
```

### 13.2 运行时执行链路

```
应用层 (PyTorch)
  │ cudaLaunchKernel() / cublasSgemm_v2() / cublasLtMatmul()
  ▼
拦截层 (LD_PRELOAD)
  │ 创建 OperationRecord, 填充参数
  │ enqueue_operation() → 放入客户端队列
  │ wait_operation() → 忙等完成
  ▼
调度器线程 (Scheduler::run())
  │ peek() → 查看队首操作
  │ 判断是否执行 (HP 直接执行, BE 看 HP 队列)
  │
  ├── [Green Context 启用 && kernel 操作]
  │     ├── cuCtxSetCurrent(cuda_ctxs_[ctx_idx])  → 切换 context
  │     ├── cudaSetDevice()
  │     └── execute_operation(op, stream, cublas_handles_[ctx_idx])
  │           └── execute_cublas_operation(op, stream, provided_handle)
  │                 ├── handle = provided_handle (Green Context handle)
  │                 ├── cublasSetStream(handle, stream)
  │                 └── cublasSgemm_v2(handle, ...) → 在隔离 SM 上执行
  │
  └── [Green Context 未启用]
        └── execute_operation(op, stream, nullptr)
              └── execute_cublas_operation(op, stream, nullptr)
                    ├── handle = thread_local_handle / original_handle
                    └── cublasSgemm_v2(handle, ...) → 在共享 SM 上执行
```

### 13.3 销毁链路

```
orion_stop_scheduler()
  └── stop_scheduler()
        ├── Scheduler::stop()     → running_ = false
        ├── Scheduler::join()     → 等待线程退出
        ├── Scheduler::reset()
        │     └── destroy_streams()
        │           ├── [Green Context] destroy_green_contexts()
        │           │     ├── cublasDestroy() ×2
        │           │     └── cuGreenCtxDestroy() ×2  (stream 随之销毁)
        │           └── [Fallback] cudaStreamDestroy() ×N
        └── shutdown_capture_layer()
```

---

## 附录 A: CUDA Driver API 调用清单

| API | 用途 | 调用位置 |
|-----|------|---------|
| `cuInit(0)` | 初始化 Driver API | `init_green_contexts()` |
| `cuDeviceGet()` | 获取设备句柄 | `init_green_contexts()` |
| `cuDeviceGetDevResource()` | 获取全部 SM 资源 | `init_green_contexts()` |
| `cuDevSmResourceSplitByCount()` | 按数量分割 SM 资源 | `init_green_contexts()` |
| `cuDevResourceGenerateDesc()` | 生成资源描述符 | `init_green_contexts()` |
| `cuGreenCtxCreate()` | 创建 Green Context | `init_green_contexts()` |
| `cuCtxFromGreenCtx()` | 派生标准 CUcontext | `init_green_contexts()` |
| `cuGreenCtxStreamCreate()` | 在 Green Context 中创建 Stream | `init_green_contexts()` |
| `cuCtxSetCurrent()` | 切换当前线程的 context | `run()` 主循环 |
| `cuGreenCtxDestroy()` | 销毁 Green Context | `destroy_green_contexts()` |
| `cuGetErrorString()` | 获取错误信息 | `CHECK_CU` 宏 |

## 附录 B: 错误处理宏

```cpp
#define CHECK_CU(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        LOG_ERROR("CUDA Driver API error at %s:%d: %s", \
                  __FILE__, __LINE__, errStr); \
        return false; \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA Runtime API error at %s:%d: %s", \
                  __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)
```

Driver API 错误使用 `CHECK_CU`，Runtime API 错误使用 `CHECK_CUDA`，两者都在失败时返回 `false` 触发 fallback。
