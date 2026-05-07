# Orion 调度器集成 Green Context 实现硬件级 SM 隔离 — 设计与验证方案

## 1. 问题描述：当前 SM 争抢现状

### 1.1 系统架构

当前 Orion 调度器通过 `LD_PRELOAD` 拦截 CUDA/cuBLAS/cuDNN API，将多个客户端（1 HP + N BE）的 GPU 操作重定向到调度器管理的 CUDA stream 上执行：

```
┌──────────────────────────────────────────────────────────────┐
│  Client 0 (HP)          Client 1 (BE)                       │
│  ┌─────────────┐        ┌─────────────┐                     │
│  │ PyTorch 推理 │        │ PyTorch 推理 │                     │
│  └──────┬──────┘        └──────┬──────┘                     │
│         │ cudaLaunchKernel     │ cublasSgemm                │
│         ▼                      ▼                             │
│  ┌──────────────────────────────────────────────┐           │
│  │           LD_PRELOAD 拦截层                    │           │
│  │  ┌──────────────┐  ┌──────────────┐          │           │
│  │  │ HP 操作队列    │  │ BE 操作队列    │          │           │
│  │  └──────┬───────┘  └──────┬───────┘          │           │
│  └─────────┼─────────────────┼──────────────────┘           │
│            ▼                 ▼                               │
│  ┌───────────────────────────────────────────────┐          │
│  │           调度器线程（单线程轮询）                │          │
│  │  HP kernel → 直接执行                          │          │
│  │  BE kernel → HP 队列为空时执行                  │          │
│  └──────┬────────────────────┬───────────────────┘          │
│         ▼                    ▼                               │
│   HP Stream               BE Stream                         │
│   (priority -5)           (priority -4)                     │
│         │                    │                               │
│         ▼                    ▼                               │
│  ┌──────────────────────────────────────────────┐           │
│  │        GPU  (114 SMs, 共享，无隔离)            │           │
│  │  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │           │
│  │  ┃  SM 0-113: HP 和 BE kernel 自由竞争      ┃  │           │
│  │  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │           │
│  └──────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────┘
```

**关键代码路径：**

| 文件 | 功能 |
|------|------|
| `src/cuda_intercept.cpp` | 拦截 `cudaLaunchKernel` 等 CUDA Runtime API |
| `src/cublas_intercept.cpp` | 拦截 `cublasSgemm_v2`、`cublasLtMatmul` 等 cuBLAS API |
| `src/scheduler.cpp` | 调度器主循环：轮询队列、决策、执行 |
| `src/gpu_capture.cpp` | 操作捕获层：队列管理、客户端索引 |
| `python/test_orion_blocking.py` | Python 测试：1 HP + N BE 并发推理 |

### 1.2 争抢的量化证据

通过 `nsys profile` 采集并分析 `profiles/orion_nsys.nsys-rep`，提取 GPU kernel trace 数据，得到以下争抢证据。

#### 1.2.1 Kernel SM 需求 vs GPU 总 SM 数

GPU 共有 **114 个 SM**（H100 PCIe）。模型中 591 个 kernel 的 SM 需求分布：

| SM 需求 | Kernel 数量 | 典型 kernel |
|---------|------------|-------------|
| 64 SM | 128 | `ampere_sgemm_128x128_tn` (大矩阵乘) |
| 32 SM | 193 | `ampere_sgemm_32x128_tn` (中矩阵乘) |
| 8 SM | 270 | `elementwise_kernel`、`layernorm_kernel` |

当 HP 和 BE 同时执行 `sgemm_128x128`（各需 64 SM）时，总需求 **128 SM > 114 SM**，必然争抢。

#### 1.2.2 Kernel 时间线重叠（nsys 实测数据）

从 GPU trace 中识别出 HP（Stream 14）和 BE（Stream 15）上 GEMM kernel 的时间重叠：

```
                    3498 ms                                    3515 ms
                      │                                          │
  Stream 14 (HP):     │████████ sgemm_128x128 (14.776 ms) ██████│
  Stream 15 (BE):   ██│██████████ sgemm_128x128 (17.026 ms) ████│██
                      │◄──────── 重叠 14.776 ms ────────────────►│
                      │          两个大 GEMM 在同一组 SM 上争抢     │
```

汇总统计：

| 指标 | 数值 |
|------|------|
| 总重叠 kernel 对数 | 120 对 |
| GEMM-vs-GEMM 重叠对数 | **49 对** |
| 总重叠时间 | 60.675 ms |
| GEMM-vs-GEMM 重叠时间 | **51.071 ms** |

#### 1.2.3 Kernel 耗时膨胀（争抢直接后果）

对比 Warmup 阶段（GPU 独占，仅 HP 运行）和正式并发阶段（HP + BE 同时运行）的同名 GEMM kernel 耗时：

| Kernel | 独占 (warmup) | 并发 (BE stream) | 耗时膨胀 |
|--------|--------------|-----------------|---------|
| `sgemm_128x64` (1024 blocks) | 1075.9 us | 2427.3 us | **+125.6%** |
| `sgemm_64x128_nn` (1024 blocks) | 541.1 us | 1153.8 us | **+113.2%** |
| `sgemm_128x128` (2048 blocks) | 12767.4 us | 13162.8 us | +3.1% |

小/中型 GEMM 的耗时膨胀超过 100%，说明 SM 争抢严重。大型 GEMM（2048 blocks，远超 114 SM）本身需要多 wave 执行，争抢的相对影响被分摊。

#### 1.2.4 宏观性能：HP 反而比 BE 慢

```
  BE1: 2 iters in 241.98 ms
  HP:  2 iters in 295.88 ms    ← HP 比 BE 慢 22%
  Total time:     295.88 ms
```

HP 被设计为高优先级（stream priority -5），但总时间反而最长。原因是调度器的软件策略无法阻止 GPU 硬件层面的 SM 争抢。

### 1.3 根本原因分析

当前调度器有两个层面的 SM "控制"机制，但都无法实现硬件级隔离：

**1. Stream Priority（CUDA 硬件机制）**

```cpp
// scheduler.cpp: create_streams()
cudaStreamCreateWithPriority(&hp_stream_, cudaStreamNonBlocking, highest_priority);   // -5
cudaStreamCreateWithPriority(&be_streams_[i], cudaStreamNonBlocking, be_priority);    // -4
```

Stream priority 只影响 GPU warp scheduler 在**同一个 SM 上**调度 warp 的优先级。它**不能阻止两个 stream 的 kernel 同时占据同一个 SM**。

**2. SM Threshold（纯软件变量）**

```cpp
// scheduler.cpp: orion_should_schedule()
if (g_capture_state.client_queues[0] && g_capture_state.client_queues[0]->empty()) {
    return true;  // HP 队列为空时才允许 BE 执行
}
return false;
```

`sm_threshold=30` 只用于调度器的软件判断（是否允许 BE kernel 入队执行），一旦 kernel 被 `cudaLaunchKernel` 发射到 GPU 上，**CUDA 硬件不限制它使用多少 SM**。一个 `sgemm_128x128` kernel 启动 2048 blocks，GPU 会将 blocks 分配到所有 114 个 SM 上执行。

**结论**：需要一种**硬件级** SM 分区机制，使 HP 和 BE 的 kernel 在物理上使用不同的 SM 集合，从根本上消除争抢。CUDA 12.4+ 的 Green Context 正是为此设计。

---

## 2. Green Context 设计方案

### 2.1 Green Context 机制概述

Green Context 是 CUDA 12.4 引入的 Driver API 特性（需要 Hopper sm_90+ 架构），允许将 GPU 的 SM 资源按数量切分为多个不重叠的分区，每个分区绑定一个独立的 CUcontext。在某个 Green Context 中启动的 kernel **只能使用该分区内的 SM**，由硬件强制保证。

```
传统模式:                          Green Context 模式:
┌─────────────────────────┐       ┌─────────────────────────┐
│  GPU (114 SMs)          │       │  GPU (114 SMs)          │
│  ┌───────────────────┐  │       │  ┌────────┐ ┌────────┐  │
│  │ HP GEMM: 全部 SM  │  │       │  │ HP GEMM│ │ BE GEMM│  │
│  │ BE GEMM: 全部 SM  │  │       │  │ 64 SMs │ │ 48 SMs │  │
│  │    ↕ 互相争抢      │  │       │  │  独占   │ │  独占   │  │
│  └───────────────────┘  │       │  └────────┘ └────────┘  │
│                         │       │       硬件隔离，无争抢     │
└─────────────────────────┘       └─────────────────────────┘
```

### 2.2 改造目标

在现有 Orion 调度器（`libgpu_scheduler.so`）中集成 Green Context，实现：

1. 调度器初始化时，根据环境变量配置将 GPU SM 资源切分为 HP 分区和 BE 分区
2. 为每个分区创建独立的 Green Context 和对应的 Stream
3. HP kernel 在 HP Green Context 的 stream 上执行，所有 BE kernel 共享 BE Green Context 的 stream 执行
4. 两组 kernel 在硬件层面使用不同的 SM，彻底消除争抢

### 2.3 架构设计

#### 2.3.1 改造后的整体架构

```
┌───────────────────────────────────────────────────────────────┐
│  Client 0 (HP)           Client 1 (BE)                       │
│  ┌─────────────┐         ┌─────────────┐                     │
│  │ PyTorch 推理 │         │ PyTorch 推理 │                     │
│  └──────┬──────┘         └──────┬──────┘                     │
│         │                       │                             │
│  ┌──────────────────────────────────────────────┐            │
│  │           LD_PRELOAD 拦截层（不变）             │            │
│  └──────┬───────────────────────┬───────────────┘            │
│         ▼                       ▼                             │
│  ┌───────────────────────────────────────────────┐           │
│  │           调度器线程                            │           │
│  │                                                │           │
│  │  HP kernel →  cuCtxSetCurrent(ctx_hp)          │           │
│  │               dispatch on hp_stream            │           │
│  │                                                │           │
│  │  BE kernel →  cuCtxSetCurrent(ctx_be)          │           │
│  │               dispatch on be_stream            │           │
│  └──────┬────────────────────┬───────────────────┘           │
│         ▼                    ▼                                │
│   HP Green Context       BE Green Context                    │
│   ┌──────────────┐       ┌──────────────┐                    │
│   │ CUcontext    │       │ CUcontext    │                    │
│   │ Stream (p-5) │       │ Stream (p-4) │                    │
│   │ SM: 0-63     │       │ SM: 64-111   │                    │
│   └──────┬───────┘       └──────┬───────┘                    │
│          ▼                      ▼                             │
│  ┌──────────────────────────────────────────────┐            │
│  │     GPU  (114 SMs, 硬件隔离)                   │            │
│  │  ┌──────────────────┐ ┌──────────────────┐   │            │
│  │  │ HP 分区: 64 SMs  │ │ BE 分区: 48 SMs  │   │            │
│  │  │ 物理隔离          │ │ 物理隔离          │   │            │
│  │  └──────────────────┘ └──────────────────┘   │            │
│  └──────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────┘
```

#### 2.3.2 SM 分区策略

Hopper 架构约束：
- 最小分配单位：**8 SM**（1 个 GPC）
- SM 数量必须对齐到 8 的整数倍
- 两个分区之和不超过总 SM 数

推荐分区方案（H100 PCIe, 114 SMs）：

| 方案 | HP SMs | BE SMs | 剩余 | 适用场景 |
|------|--------|--------|------|---------|
| HP 优先 | 72 | 40 | 2 | 推理延迟敏感（默认） |
| 均衡 | 56 | 56 | 2 | HP/BE 同等重要 |
| BE 优先 | 40 | 72 | 2 | 训练吞吐优先 |

通过环境变量 `ORION_HP_SMS` 和 `ORION_BE_SMS` 在启动时配置，无需运行时动态调整。

### 2.4 代码改造详细设计

#### 2.4.1 新增数据结构 — `include/scheduler.h`

在 `Scheduler` 类中新增 Green Context 相关成员：

```cpp
#include <cuda.h>  // CUDA Driver API

class Scheduler {
public:
    // ... 现有接口不变 ...

    // Green Context SM 配置（启动时从环境变量读取）
    struct GreenCtxConfig {
        unsigned int hp_sm_count = 64;
        unsigned int be_sm_count = 48;
        bool enabled = false;
    };

    bool init_green_contexts(const GreenCtxConfig& gc_config);
    void destroy_green_contexts();

private:
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
```

#### 2.4.2 Green Context 初始化 — `src/scheduler.cpp`

在 `create_streams()` 中根据环境变量决定是否启用 Green Context：

```cpp
static Scheduler::GreenCtxConfig read_green_ctx_config() {
    Scheduler::GreenCtxConfig gc;
    const char* hp = getenv("ORION_HP_SMS");
    const char* be = getenv("ORION_BE_SMS");
    if (hp && be) {
        gc.hp_sm_count = std::atoi(hp);
        gc.be_sm_count = std::atoi(be);
        gc.enabled = (gc.hp_sm_count > 0 && gc.be_sm_count > 0);
    }
    return gc;
}

bool Scheduler::init_green_contexts(const GreenCtxConfig& gc_config) {
    gc_config_ = gc_config;

    // Step 1: 初始化 CUDA Driver API
    CHECK_CU(cuInit(0));
    CHECK_CU(cuDeviceGet(&cu_device_, config_.device_id));

    // Step 2: 获取全部 SM 资源
    CHECK_CU(cuDeviceGetDevResource(
        cu_device_, &full_sm_resource_, CU_DEV_RESOURCE_TYPE_SM));

    // Step 3: 对齐 SM 数量到 8 的倍数
    const unsigned unit = 8;
    unsigned int hp_sms = ((gc_config.hp_sm_count + unit - 1) / unit) * unit;
    unsigned int be_sms = ((gc_config.be_sm_count + unit - 1) / unit) * unit;

    if (hp_sms + be_sms > (unsigned)config_.num_sms) {
        LOG_ERROR("SM partition exceeds total: HP(%u) + BE(%u) > %d",
                  hp_sms, be_sms, config_.num_sms);
        return false;
    }

    LOG_INFO("SM partition: HP=%u, BE=%u, remaining=%d",
             hp_sms, be_sms, config_.num_sms - hp_sms - be_sms);

    // Step 4: 分割 SM 资源为两组
    CUdevResource remaining{};
    unsigned int hp_groups = 1;
    CHECK_CU(cuDevSmResourceSplitByCount(
        &sm_groups_[0], &hp_groups, &full_sm_resource_, &remaining,
        0, hp_sms));

    unsigned int be_groups = 1;
    CHECK_CU(cuDevSmResourceSplitByCount(
        &sm_groups_[1], &be_groups, &remaining, nullptr,
        0, be_sms));

    // Step 5: 创建 Green Context
    for (int i = 0; i < 2; i++) {
        CHECK_CU(cuDevResourceGenerateDesc(&sm_descs_[i], &sm_groups_[i], 1));
        CHECK_CU(cuGreenCtxCreate(&green_ctxs_[i], sm_descs_[i],
                                   cu_device_, CU_GREEN_CTX_DEFAULT_STREAM));
        CHECK_CU(cuCtxFromGreenCtx(&cuda_ctxs_[i], green_ctxs_[i]));
    }

    // Step 6: 在各自 Green Context 中创建 Stream 和 cuBLAS Handle
    // HP
    CHECK_CU(cuCtxSetCurrent(cuda_ctxs_[0]));
    CUstream cu_hp_stream;
    CHECK_CU(cuGreenCtxStreamCreate(&cu_hp_stream, green_ctxs_[0],
                                     CU_STREAM_NON_BLOCKING, 0));
    hp_stream_ = reinterpret_cast<cudaStream_t>(cu_hp_stream);
    cublasCreate(&cublas_handles_[0]);
    LOG_INFO("HP Green Context: %u SMs, stream + cuBLAS handle created", hp_sms);

    // BE
    CHECK_CU(cuCtxSetCurrent(cuda_ctxs_[1]));
    CUstream cu_be_stream;
    CHECK_CU(cuGreenCtxStreamCreate(&cu_be_stream, green_ctxs_[1],
                                     CU_STREAM_NON_BLOCKING, 0));
    be_streams_.resize(1);
    be_streams_[0] = reinterpret_cast<cudaStream_t>(cu_be_stream);
    cublasCreate(&cublas_handles_[1]);
    LOG_INFO("BE Green Context: %u SMs, stream + cuBLAS handle created", be_sms);

    green_ctx_initialized_ = true;
    return true;
}
```

#### 2.4.3 调度器执行时切换 Context — `src/scheduler.cpp`

修改 `run()` 主循环，在执行 kernel 前切换到对应的 Green Context：

```cpp
void Scheduler::run() {
    tl_is_scheduler_thread = true;
    LOG_INFO("Scheduler thread running (single-threaded polling)");

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

                // ★ 关键改动：执行前切换到对应的 Green Context
                int ctx_idx = (i == 0) ? 0 : 1;
                if (green_ctx_initialized_ && is_kernel_operation(op->type)) {
                    cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
                }

                cudaError_t err = execute_operation(op, stream,
                    green_ctx_initialized_ ? cublas_handles_[ctx_idx] : nullptr);

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
    // ... 剩余操作处理同前 ...
}
```

#### 2.4.4 SM 隔离验证工具 — `tests/test_sm_isolation.cu`（独立测试程序，不编入生产 .so）

开发阶段用于一次性验证 Green Context 的 SM 分区是否正确，验证通过后不再需要：

```cuda
#include <cstdio>
#include <set>
#include <vector>
#include <cuda_runtime.h>

__global__ void probe_sm_ids(int* d_sm_ids, int num_blocks) {
    if (threadIdx.x == 0 && blockIdx.x < num_blocks) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        d_sm_ids[blockIdx.x] = static_cast<int>(smid);
    }
}

std::set<int> probe_sm_set(cudaStream_t stream, int num_blocks) {
    int* d_sm_ids = nullptr;
    cudaMalloc(&d_sm_ids, num_blocks * sizeof(int));
    cudaMemset(d_sm_ids, 0xff, num_blocks * sizeof(int));

    probe_sm_ids<<<num_blocks, 1, 0, stream>>>(d_sm_ids, num_blocks);
    cudaStreamSynchronize(stream);

    std::vector<int> h_sm_ids(num_blocks);
    cudaMemcpy(h_sm_ids.data(), d_sm_ids,
               num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_sm_ids);

    std::set<int> sm_set(h_sm_ids.begin(), h_sm_ids.end());
    sm_set.erase(-1);
    return sm_set;
}
```

编译方式：`nvcc -arch=sm_90 -o test_sm_isolation tests/test_sm_isolation.cu -lcuda`，独立于主 Makefile。

#### 2.4.5 配置方式 — 环境变量

SM 分区在调度器启动时通过环境变量一次性配置，不提供运行时动态调整接口（Green Context 分区变更需要销毁重建，运行时调整意义不大）：

```bash
# 启用 Green Context，HP 64 SM，BE 48 SM
ORION_HP_SMS=64 ORION_BE_SMS=48 LD_PRELOAD=build/libgpu_scheduler.so python3 test.py

# 不设置环境变量则自动 fallback 到 stream priority 方案
LD_PRELOAD=build/libgpu_scheduler.so python3 test.py
```

#### 2.4.6 Python 测试脚本修改 — `python/test_orion_blocking.py`

Green Context 的启用完全由环境变量控制，Python 测试脚本无需修改 C 接口调用。调度器在 `create_streams()` 中自动读取 `ORION_HP_SMS` / `ORION_BE_SMS` 环境变量并初始化。

测试时只需在命令行指定环境变量：

```bash
# 带 Green Context
ORION_HP_SMS=64 ORION_BE_SMS=48 \
  LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py

# 不带 Green Context（对照组）
LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py
```

#### 2.4.7 Makefile 修改

新增 CUDA Driver API 链接（SM 探测工具独立编译，不影响主 Makefile）：

```makefile
# 新增链接 CUDA Driver API (-lcuda)
LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lcuda -lnvToolsExt
```

### 2.5 对现有代码的影响范围

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `include/scheduler.h` | 修改 | 新增 `GreenCtxConfig`、Green Context 成员变量、`cublas_handles_[2]` |
| `src/scheduler.cpp` | 修改 | 新增 `init_green_contexts()`、`destroy_green_contexts()`、修改 `run()` 加入 `cuCtxSetCurrent` 和 cuBLAS handle 传递 |
| `tests/test_sm_isolation.cu` | **新增** | 独立 SM 隔离验证工具（不编入 .so） |
| `python/test_orion_blocking.py` | 不变 | Green Context 由环境变量控制，脚本无需改动 |
| `Makefile` | 修改 | 确保链接 `-lcuda` |
| `include/gpu_capture.h` | 不变 | 操作捕获层不涉及 SM 分区 |
| `src/cuda_intercept.cpp` | 不变 | 拦截逻辑不变，操作仍提交到队列 |
| `src/cublas_intercept.cpp` | 不变 | cuBLAS 拦截不变 |

**核心设计原则**：拦截层（capture layer）不变，只改调度器（scheduler）的 stream 创建和 kernel 执行方式。客户端提交操作的流程完全不变，改动被封装在调度器内部。

### 2.6 cuBLAS Handle 的 Context 亲和性处理

cuBLAS 的 handle 通过 `cublasCreate` 绑定到创建时的 CUDA context。在 Green Context 方案中，初始化时为每个 context 创建独立的 cuBLAS handle，存储在 `cublas_handles_[2]` 数组中（见 2.4.2 Step 6）。

调度器是单线程轮询模型，执行 cuBLAS 操作时根据 `ctx_idx`（0=HP, 1=BE）直接索引对应的 handle，无需加锁：

```cpp
// 在 execute_operation() 中使用对应 context 的 cuBLAS handle
cublasHandle_t handle = cublas_handles_[ctx_idx];
cublasSetStream(handle, stream);
// ... 执行 cuBLAS 操作 ...
```

### 2.7 `cuCtxSetCurrent` 的性能开销

`cuCtxSetCurrent()` 是线程局部操作，只修改调用线程的 context 指针，**不涉及 GPU 端同步**。实测开销在微秒级，相比 GEMM kernel 的毫秒级执行时间可忽略不计。

从 `green_ctx_per_operator_design.md` 的实验数据：

```
Context switch overhead: 0.005 ms
```

调度器主循环每次迭代最多切换一次 context，额外开销约 5 us，对总延迟影响 < 0.01%。

---

## 3. 验证方案

### 3.1 SM 隔离正确性验证（一次性）

使用独立测试工具 `tests/test_sm_isolation.cu` 验证两个 Green Context 的 SM 集合不重叠。开发阶段运行一次，确认硬件隔离生效即可。

```bash
# 编译并运行 SM 探测
nvcc -arch=sm_90 -o test_sm_isolation tests/test_sm_isolation.cu -lcuda
ORION_HP_SMS=64 ORION_BE_SMS=48 ./test_sm_isolation
```

预期输出：

```
HP SMs: { 0 1 2 ... 63 } (count=64)
BE SMs: { 64 65 66 ... 111 } (count=48)
Disjoint: YES
```

### 3.2 端到端性能对比

运行两组实验对比 HP 延迟：

```bash
# A. Baseline（无 Green Context）
LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py

# B. Green Context
ORION_HP_SMS=64 ORION_BE_SMS=48 \
  LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py
```

通过标准：
- Green Context 下 HP 延迟 < Baseline HP 延迟
- HP 延迟 ≤ Solo 延迟 × (114/HP_SMs) × 1.1
- cuBLAS GEMM 结果正确（与 Solo 一致）

### 3.3 nsys 深度分析（调优阶段，可选）

功能验证通过后，可使用 nsys 采集 kernel trace 进一步分析争抢消除效果：

```bash
nsys profile --trace=cuda,nvtx --output=profiles/green_ctx_nsys \
  --force-overwrite=true \
  env ORION_HP_SMS=64 ORION_BE_SMS=48 \
  LD_PRELOAD=build/libgpu_scheduler.so python3 python/test_orion_blocking.py

nsys stats --report cuda_gpu_trace --format csv \
  --output profiles/green_ctx_trace.csv profiles/green_ctx_nsys.nsys-rep

python3 scripts/analyze_sm_contention.py profiles/green_ctx_trace.csv_cuda_gpu_trace.csv
```

关注指标：中型 GEMM 并发膨胀率应 < 15%（Baseline 为 >100%）。

---

## 4. 环境要求与风险

### 4.1 环境要求

| 要求 | 当前值 | 最低要求 |
|------|-------|---------|
| CUDA Toolkit | 12.8 | 12.4+ |
| GPU 架构 | H100 (sm_90) | Hopper (sm_90+) |
| Driver | - | 550.54+ |
| Green Context API | 可用 | `cuGreenCtxCreate` 等 |

### 4.2 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| Green Context API 不可用（旧 GPU/Driver） | 功能退化 | 保留 fallback：检测失败时回退到现有 stream priority 方案 |
| SM 分区后单客户端性能下降 | HP 延迟可能高于 Solo | 这是预期行为（SM 减少）；通过调整环境变量比例优化 |
| cuBLAS handle 绑定 context | GEMM 执行错误 | 初始化时为每个 Green Context 创建独立的 handle（`cublas_handles_[2]`） |
| `cuCtxSetCurrent` 线程安全 | 调度器是单线程 | 单线程 polling 模型天然避免竞争 |

### 4.3 Fallback 策略

```cpp
bool Scheduler::create_streams() {
    auto gc_config = read_green_ctx_config();  // 从环境变量读取
    if (gc_config.enabled) {
        bool ok = init_green_contexts(gc_config);
        if (ok) {
            LOG_INFO("Green Context SM isolation enabled");
            return true;
        }
        LOG_WARN("Green Context init failed, falling back to stream priority");
    }
    // 现有逻辑：cudaStreamCreateWithPriority
    return create_priority_streams();
}
```

---

## 5. 实施路线

| 阶段 | 内容 | 产出 |
|------|------|------|
| Phase 1 | 在 `scheduler.cpp` 中实现 `init_green_contexts()`：环境变量读取、SM 分区、Green Context 创建、Stream + cuBLAS handle 创建；修改 `run()` 加入 `cuCtxSetCurrent` 切换 | 编译通过，端到端测试通过 |
| Phase 2 | 编写独立 SM 探测工具 `tests/test_sm_isolation.cu`，验证 SM 隔离正确性 | 输出 "Disjoint: YES" |
| Phase 3 | 对比 Baseline 和 Green Context 的 HP 延迟，可选 nsys 深度分析 | 性能数据确认改善 |
