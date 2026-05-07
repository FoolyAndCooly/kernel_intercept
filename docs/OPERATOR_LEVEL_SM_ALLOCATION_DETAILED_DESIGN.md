# 算子粒度 SM 资源分配：动态 Green Context 切换设计

## 1. 问题背景

### 1.1 核心矛盾

当前静态 Green Context 方案（HP 固定 56 SMs，BE 固定 56 SMs）存在问题：

- **无争用时**：HP 和 BE 各自被限制在 56 SMs，单任务无法利用全部 GPU 资源

**已被验证虽然可以并发，但是无法得到性能提升，静态Green Context方案均存在一致的问题，有必要尝试更高风险和收益的模式切换方案**

理想状态是：**无争用时不隔离，有争用时自动隔离**。

## 2. 共同主干架构

所有方案共享同一套核心架构，差异仅在于**何时触发模式切换**。

### 2.1 两种执行模式

```
DEFAULT 模式（SM 需求无冲突）    ISOLATED 模式（SM 需求冲突）
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ HP Stream  │  │ BE Stream  │  │ HP GC      │  │ BE GC      │
│ hp_stream_ │  │ be_stream_ │  │ hp_gc_stream│  │ be_gc_stream│
│ hp_cublas  │  │ be_cublas  │  │ hp_cublas  │  │ be_cublas  │
│ 114 SMs    │  │ 114 SMs    │  │ 56 SMs     │  │ 56 SMs     │
│ 共享全量   │  │ 共享全量   │  │ HP 专用    │  │ BE 专用    │
└────────────┘  └────────────┘  └────────────┘  └────────────┘
   并发执行，自由竞争 SM           并发执行，SM 隔离互不干扰
```

**说明**：
- **DEFAULT 模式**：HP 和 BE 各有独立 stream，并发执行，共享全量 114 SMs，自由竞争。适用于 SM 需求总和 ≤ 114 的场景（如一个大 kernel + 一个小 kernel，或两个中等 kernel）
- **ISOLATED 模式**：HP 和 BE 各有独立 GC stream，并发执行，SM 硬隔离。**当前 CUDA 12.8 实现默认使用 HP 56 SMs + BE 56 SMs**，优先保证 Green Context 能稳定初始化；更激进的异构切分（如 64/48）保留为后续升级方向。适用于 SM 需求总和 > 114 的场景（如两个大 kernel 同时运行，会严重争用）
- **切换触发条件**：检测到 `total_sm_demand > 114` 时切换到 ISOLATED，否则保持 DEFAULT
- SM 数量基于 H100 PCIe（114 SMs），当前实现默认 HP 56 + BE 56 = 112，剩余 2 SMs

### 2.2 资源初始化

两套资源在启动时同时创建，DEFAULT 资源立即使用，GC 资源预创建备用：

```cpp
struct StreamSet {
    // DEFAULT 模式资源（两个独立 stream，无 GC 隔离）
    cudaStream_t hp_stream  = nullptr;
    cudaStream_t be_stream  = nullptr;
    cublasHandle_t hp_cublas_default = nullptr;
    cublasHandle_t be_cublas_default = nullptr;

    // ISOLATED 模式资源（预创建，按需激活）
    CUgreenCtx hp_green_ctx = nullptr;
    CUgreenCtx be_green_ctx = nullptr;
    CUcontext  hp_cuda_ctx  = nullptr;
    CUcontext  be_cuda_ctx  = nullptr;
    cudaStream_t hp_gc_stream = nullptr;
    cudaStream_t be_gc_stream = nullptr;
    cublasHandle_t hp_cublas_gc  = nullptr;
    cublasHandle_t be_cublas_gc  = nullptr;
};
```

SM 分区设计上优先希望支持异构分区，但**当前机器为 CUDA 12.8，实际实现使用 Driver API 的 `cuDevSmResourceSplitByCount` 单次切出两个 56-SM 同构分组**：

```cpp
CUdevResource remaining{};
unsigned int group_count = 2;
cuDevSmResourceSplitByCount(sm_groups_, &group_count,
                            &full_sm_resource_, &remaining,
                            0, 56);
```

> **API 说明**：当前实现使用 CUDA Driver API（`CUgreenCtx`、`cuGreenCtxCreate`、`cuGreenCtxStreamCreate`、`cuCtxSetCurrent`）。CUDA 4.6 规范推荐使用 Runtime API，原因有二：
> 1. **Work Queue 隔离**：Runtime API 的 `cudaGreenCtxCreate` 有 `numExpectedConcurrentKernels` 参数，driver 用此提示避免 HP/BE 共享 work queue。Driver API 的 `cuGreenCtxCreate` 没有此参数，**work queue 隔离无法配置**，即使 SM 已隔离，HP/BE 仍可能因共享 work queue 而被串行化。
> 2. **Context 管理简化**：Runtime API stream 创建时已绑定 execution context，无需手动 `cuCtxSetCurrent`，也不存在 primary context 恢复问题。
>
> 但本机 CUDA 版本为 **12.8**，尚未具备文档中 CUDA 13.1 Runtime execution context 那套推荐接口，因此当前版本继续采用 Driver API 落地。与此同时，当前实现路径选择了最稳妥的同构分割方式：**单次 `cuDevSmResourceSplitByCount(..., nbGroups=2, minCount=56)` 切出两个 56-SM 分组**。理论上的 64/48 异构切分仍保留为后续升级方向；如需 work queue 隔离与异构分区能力，后续需升级到更新的 Runtime API 路线。

### 2.3 模式切换机制

切换时必须等待旧 stream 上的 kernel 完成，否则旧 kernel 和新 kernel 会同时占用 SM，隔离失效。

```cpp
void Scheduler::switch_mode(ExecutionMode new_mode) {
    if (new_mode == current_mode_) return;

    // 等待当前模式的所有 stream 完成，确保 SM 资源释放
    if (current_mode_ == DEFAULT) {
        cudaStreamSynchronize(hp_stream_);
        cudaStreamSynchronize(be_stream_);
    } else {
        cudaStreamSynchronize(hp_gc_stream_);
        cudaStreamSynchronize(be_gc_stream_);
    }

    // ISOLATED → DEFAULT：必须恢复 primary context
    if (new_mode == DEFAULT && current_mode_ == ISOLATED) {
        cuCtxSetCurrent(primary_ctx_);
    }

    current_mode_ = new_mode;
    current_ctx_idx_ = -1;  // 重置 context 缓存
}
```

`primary_ctx_` 需要在初始化时保存（GC 初始化之前）：
```cpp
cuDevicePrimaryCtxRetain(&primary_ctx_, cu_device_);
```

切换开销主要来自等待当前 kernel 完成，而非切换操作本身（`cuCtxSetCurrent` 约 5-20μs）。

### 2.3.1 切换开销量化

切换开销 = **等待当前模式所有 stream 上的 kernel 完成的时间**，与 kernel 的剩余执行时间直接相关。

**两个方向的开销不对称：**

| 切换方向 | 等待对象 | 典型开销 | 触发条件 |
|---------|---------|---------|---------|
| DEFAULT → ISOLATED | hp_stream_ + be_stream_ 上的 kernel | 0 ~ T_kernel（取决于检测时机） | 检测到 SM 争用 |
| ISOLATED → DEFAULT | hp_gc_stream_ + be_gc_stream_ 上的 kernel | 0 ~ T_kernel | 争用消失（冷却期结束） |

**关键约束**：`cudaStreamSynchronize` 是阻塞调用，调度器线程在等待期间无法处理任何新操作。等待期间新到达的操作只能积压在队列中。

### 2.4 调度主循环骨架

```cpp
void Scheduler::run() {
    while (running_) {
        // ① 前瞻决策：扫描未来算子序列判断是否需要切换模式
        ExecutionMode desired = decide_mode();
        if (desired != current_mode_) {
            switch_mode(desired);
        }

        // ② 执行：两种模式下 HP 和 BE 均并发，区别仅在于 stream 是否绑定 GC
        for (int i = 0; i < num_clients_; i++) {
            auto op = g_capture_state.client_queues[i]->peek();
            if (!op) continue;

            g_capture_state.client_queues[i]->try_pop();
            // 根据 current_mode_ 选择 stream/handle/context
            dispatch_in_current_mode(op, i);
        }
    }
}
```

---

## 3. 算子序列前瞻策略 (Operator Sequence Lookahead)

### 3.1 核心洞察：算子顺序固定且完全可预测

Deep learning workload（推理/训练）的算子提交顺序由模型的计算图决定。在相同输入形状下，每次迭代产生**完全相同的 CUDA kernel 序列**——算子类型、SM 需求、执行时长的顺序都固定不变。

当前系统已经具备利用这一特性的全部基础设施：

1. **`kernel_info.csv`** 存储了每个客户端的完整算子序列（名称、SM 需求、执行时长）
2. **`g_orion_state.op_info_vector[i]`** 在启动时加载该序列
3. **`g_orion_state.seen[i]`** 作为**程序计数器 (PC)** 跟踪每个客户端的执行进度
4. **`seen[i] % profile.size()`** 使序列在每次迭代中循环重复

这意味着调度器拥有的不是概率性估计，而是**确定性知识**：

```
                     seen[i]
                       ↓
profile[i]: [ K0, K1, K2, K3, K4, K5, K6, K7, ... ]
                       ^当前  ^+1   ^+2  ^+3
                       |      |     |    |
                       知道   知道  知道  知道
```

不仅知道当前算子（`seen[i]`），还知道未来所有算子（`seen[i]+1`, `seen[i]+2`, ...）的 SM 需求和执行时长。这是相比反应式策略（只看"GPU 上当前在跑什么"）和统计预测（基于历史频率估算）的根本优势。

### 3.2 方向一：前瞻冲突预测 (Lookahead Conflict Prediction)

**核心思想**：在 dispatch 当前算子时，同时查看 `seen[i] + K` 位置的算子 SM 需求，提前预判未来是否存在 SM 冲突。

#### 3.2.1 max-in-window 策略（当前实现）

**关键设计**：不采用步对齐比较（在相同 step 偏移处比较各客户端的 SM 需求），而是取每个客户端在前瞻窗口内的**最大 SM 需求**，再求和判断是否超限。

**步对齐比较的缺陷**：初始版本在 `step=0,1,...,K` 处逐步比较 `sum(peek_sm(c, step))`，隐含假设各客户端在相同 step 偏移处的 kernel 会在 GPU 上同时执行。但实际上：

1. HP 和 BE 的 `seen[]` 计数器独立漂移（HP 始终先调度，BE 可能被延迟），导致同一 step 偏移并不对应同一时刻
2. 长短 kernel 的时长差异导致不同 step 的 kernel 在 GPU 上时间重叠。例如 HP 的 `ampere_sgemm_128x128_nn`（64SM, ~2.6ms）执行期间，BE 可跨越 3 个 `ampere_sgemm_32x128_tn`（32SM, ~0.4ms）后到达自己的 64SM kernel

实测表现：SM 序列为 `...32,32,32,64,8,8,8,8,64,...`，步对齐比较在每个 step 上看到的是 `64+32=96` 或 `32+64=96`（均 ≤ 114），**永远检测不到 64+64=128 的冲突**，导致 switches=0。

```cpp
bool Scheduler::predict_sm_conflict() {
    std::vector<int> active_clients;
    for (int i = 0; i < num_clients_; i++) {
        if (!g_capture_state.client_queues[i]) continue;
        auto op = g_capture_state.client_queues[i]->peek();
        if (op && is_kernel_operation(op->type)) {
            active_clients.push_back(i);
        }
    }
    if ((int)active_clients.size() < 2) return false;

    // max-in-window：取每个客户端在窗口内的峰值 SM，再求和
    int total_max_sm = 0;
    for (int c : active_clients) {
        int max_sm = 0;
        for (int step = 0; step <= lookahead_window_; step++) {
            max_sm = std::max(max_sm, peek_sm_requirement(c, step));
        }
        total_max_sm += max_sm;
    }

    if (total_max_sm > config_.num_sms) {
        return true;  // 窗口内峰值 SM 总和超限
    }
    return false;
}
```

**max-in-window 的正确性保证**：

| 场景 | 步对齐结果 | max-in-window 结果 | 实际 GPU |
|------|-----------|-------------------|---------|
| HP=[64,32,8], BE=[32,64,8] | step0: 96≤114, step1: 96≤114, step2: 16≤114 → 无冲突 | max(64,32,8)+max(32,64,8) = 64+64 = 128>114 → **冲突** | 64SM HP kernel 跨越 BE 的短 kernel 后与 BE 的 64SM 重叠 → **冲突** |
| HP=[8,8,8], BE=[8,8,8] | 16≤114 → 无冲突 | 8+8=16≤114 → 无冲突 | 无冲突 |
| HP=[64,8,8], BE=[8,8,8] | 72≤114 → 无冲突 | 64+8=72≤114 → 无冲突 | 无冲突 |

max-in-window 是保守策略：可能存在误报（两个客户端的峰值 kernel 不一定在时间上重叠），但不会漏报。对于模式切换的场景，误报（多切一次）的代价远小于漏报（64+64 争用不隔离）。

**前瞻窗口 `lookahead_window_` 的选择**：

- 默认值 `3`：覆盖 transformer 中一个 attention block 的核心算子跨度（通常包含 LayerNorm → QKV GEMM → Attention → Projection → FFN）
- 窗口太小（1）：退化为"只看当前"，失去前瞻优势
- 窗口太大（>10）：可能看到下一轮迭代的算子，引入噪声（因为两个客户端的相位可能在迭代边界发生变化）；同时 max-in-window 的保守性会随窗口增大而增加误报率
- 可通过 `ORION_LOOKAHEAD_WINDOW` 环境变量调整

### 3.3 方向二：窗口争用密度扫描 (Window Conflict Ratio)

前瞻冲突预测只回答"有没有冲突"，争用密度扫描回答"冲突有多密集"。这在 **ISOLATED → DEFAULT** 切换时尤为重要：冷却期到了，但如果未来仍然冲突密集，贸然退出 ISOLATED 会导致立刻再切回来。

#### 3.3.1 进入/退出策略的不对称性

**关键设计原则**：进入和退出使用不同保守度的冲突检测策略。

| 方向 | 函数 | 策略 | 原因 |
|------|------|------|------|
| DEFAULT → ISOLATED | `predict_sm_conflict()` | max-in-window（保守） | 漏检代价高：64+64 碰撞导致严重争用 |
| ISOLATED → DEFAULT | `lookahead_conflict_ratio()` | 步对齐（宽松） | 误锁代价高：永久 56SM 限制损失 > 偶尔碰撞 |

**教训**：当 64SM kernel 密度高时（如每 4-5 个位置出现一次，占 22%），max-in-window 的子窗口（4 个连续位置）有 89% 概率命中 64SM。两个客户端独立命中后 ratio 达 83%，永远超过 30% 阈值，系统锁死在 ISOLATED 模式。实测表明永久 ISOLATED（307ms）比全程 DEFAULT（298ms）**慢 3%**，因为 56SM 限制对 78% 的小 kernel（8/32 SM）造成的性能损失远大于偶尔 64+64 碰撞的损害。

```cpp
float Scheduler::lookahead_conflict_ratio(int window_size) {
    int conflict_count = 0;
    for (int step = 0; step < window_size; step++) {
        int total_sm = 0;
        int active = 0;
        for (int i = 0; i < num_clients_; i++) {
            int sm = peek_sm_requirement(i, step);
            if (sm > 0) {
                total_sm += sm;
                active++;
            }
        }
        if (active >= 2 && total_sm > config_.num_sms) {
            conflict_count++;
        }
    }
    return (window_size > 0) ? (float)conflict_count / window_size : 0.0f;
}
```

步对齐在此处的优势：只有当两个客户端的 `seen[]` 偏移恰好使 64SM kernel 对齐到同一步时才报告冲突。对于本 profile（周期 9，64SM 在位置 3 和 8），步对齐冲突率约 0-22%（取决于偏移量 d%9），绝大多数情况下低于 30% 阈值，允许系统退回 DEFAULT。

**在 `decide_mode()` 中的用法**：

```cpp
// 冷却期已过，但检查扩展前瞻窗口
float ratio = lookahead_conflict_ratio(lookahead_window_ * 2);
if (ratio > 0.3f) {
    // 步对齐也显示高冲突 → 两个客户端 64SM kernel 真正对齐，继续隔离
    return ExecutionMode::ISOLATED;
}
// 步对齐冲突率低 → 64SM kernel 不对齐，DEFAULT 更优
```

这个机制替代了旧策略 D（纯时间冷却期）的"盲等"模式：不再是"等够 50ms 就退出"，而是"等够冷却期 **且** 未来看起来安全才退出"。

### 3.4 方向三：最优切换点对齐 (Optimal Switch Point Alignment)

模式切换的代价是 `cudaStreamSynchronize`，等待时间等于**当前 stream 上最后一个 kernel 的剩余执行时间**。既然知道完整的算子序列，就可以选择一个"切换代价最小"的时刻——即在一个短 kernel 刚被 dispatch 后触发切换。

```cpp
bool Scheduler::is_good_switch_point() {
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);
    for (int i = 0; i < num_clients_; i++) {
        if (i >= (int)g_orion_state.op_info_vector.size()) continue;
        auto& profile = g_orion_state.op_info_vector[i];
        if (profile.empty()) continue;

        int seen = g_orion_state.seen[i];
        if (seen == 0) continue;  // 尚未 dispatch 任何 kernel，stream 空闲

        int prev_idx = (seen - 1) % (int)profile.size();
        if (profile[prev_idx].duration > switch_duration_threshold_ns_) {
            return false;  // 上一个 kernel 是长算子，stream 可能还在执行
        }
    }
    return true;
}
```

**Transformer 模型中的天然切换点**：

```
Block 结构: [LayerNorm] [QKV GEMM] [Attention] [Proj GEMM] [LayerNorm] [FFN GEMM] [FFN GEMM]
               短 ~10μs   长 ~2ms    长 ~3ms    长 ~1ms      短 ~10μs   长 ~2ms    长 ~1ms
               ↑ 好切换点                                     ↑ 好切换点
```

LayerNorm、Dropout、Activation 等小算子的 duration 通常在 10-50μs 级别，在这些点切换时 `cudaStreamSynchronize` 的等待时间可以忽略不计。

**当前实现策略**：

- `is_good_switch_point()` 用于 **ISOLATED → DEFAULT** 切换：冷却期已过且未来安全时，等待一个好的切换点再真正退出
- **DEFAULT → ISOLATED** 切换不检查切换点，立即切换：因为争用损失（大 kernel 互相挤压 SM）通常远大于切换等待的开销

### 3.5 方向四：HP/BE 算子相位分析 (Phase Alignment Analysis)

由于 HP 和 BE 通常运行相同或相似的模型，它们的算子序列是相同的。两个序列之间的"相位差"（`seen[0]` 和 `seen[1]` 的差值）决定了冲突模式：

```
HP:  [LN] [GEMM] [Attn] [GEMM] [LN] [FFN] [FFN] [LN] [GEMM] ...
BE:  [LN] [GEMM] [Attn] [GEMM] [LN] [FFN] [FFN] [LN] [GEMM] ...

相位对齐（seen 差值小）：
HP:  [LN] [GEMM] [Attn] [GEMM]     ← 大 kernel 重叠，冲突密集
BE:       [GEMM] [Attn] [GEMM]

相位错开（seen 差值约半个 block）：
HP:  [LN] [GEMM] [Attn] [GEMM]     ← 大小 kernel 交错，天然互补
BE:                          [LN] [FFN] [FFN] [LN]
```

**利用方式**：

- 计算相位差 `phase_diff = abs(seen[0] - seen[1]) % profile.size()`
- 相位差接近 0 或 `profile.size()` → 高冲突风险 → 倾向 ISOLATED
- 相位差接近 `profile.size() / 2` → 天然互补 → 倾向 DEFAULT

**当前状态**：相位分析作为后续优化方向，当前实现暂不包含。原因是 `run.sh` 压测场景中 HP/BE 同时起跑，相位差在初始阶段不可控，且前瞻冲突预测已经能覆盖相位对齐的情况（两个大 GEMM 同时到达时 `predict_sm_conflict()` 会检测到）。

---

## 4. 综合决策流程

### 4.1 `decide_mode()` 完整逻辑

将前瞻冲突预测、争用密度扫描、切换点对齐三个方向组合成统一的决策函数：

```cpp
ExecutionMode Scheduler::decide_mode() {
    auto now = std::chrono::steady_clock::now();

    // ① 前瞻冲突预测：扫描当前 + 未来 K 步
    if (predict_sm_conflict()) {
        last_contention_time_ = now;
        return ExecutionMode::ISOLATED;
    }

    // ② ISOLATED → DEFAULT 的三层保护
    if (current_mode_ == ExecutionMode::ISOLATED) {
        float elapsed_ms = std::chrono::duration<float, std::milli>(
            now - last_contention_time_).count();

        // 第一层：时间冷却期
        if (elapsed_ms < cooldown_ms_) {
            return ExecutionMode::ISOLATED;
        }

        // 第二层：扩展前瞻确认未来安全
        float ratio = lookahead_conflict_ratio(lookahead_window_ * 2);
        if (ratio > 0.3f) {
            last_contention_time_ = now;
            return ExecutionMode::ISOLATED;
        }

        // 第三层：等待低代价切换点
        if (!is_good_switch_point()) {
            return ExecutionMode::ISOLATED;
        }
    }

    return ExecutionMode::DEFAULT;
}
```

**决策流程图**：

```
每轮调度循环
    │
    ▼
predict_sm_conflict()  ──── 扫描 seen[i]+0..+K 的 SM 需求
    │
    ├─ 冲突 → 立即 ISOLATED（不等切换点，争用损失 > 切换开销）
    │
    └─ 无冲突
         │
         ├─ 当前 DEFAULT → 保持 DEFAULT
         │
         └─ 当前 ISOLATED
              │
              ├─ 冷却期内 → 保持 ISOLATED
              │
              ├─ 冷却期过 + 未来冲突密 → 保持 ISOLATED
              │
              ├─ 冷却期过 + 未来安全 + 非好切换点 → 保持 ISOLATED（等切换点）
              │
              └─ 冷却期过 + 未来安全 + 好切换点 → 切换 DEFAULT
```

### 4.2 BE 调度判断 (`orion_should_schedule`)

在 DEFAULT 模式下，BE kernel 是否可以与 HP 并发执行的判断也改为使用 profile 数据：

```cpp
bool Scheduler::orion_should_schedule(OperationPtr op, int client_idx) {
    if (is_memory_operation(op->type)) return true;
    if (!is_kernel_operation(op->type)) return true;

    // ISOLATED 模式：SM 硬隔离，始终允许
    if (current_mode_ == ExecutionMode::ISOLATED) return true;

    // DEFAULT 模式：检查 HP 和 BE 的 SM 总需求
    int hp_sm = peek_sm_requirement(0);       // HP 当前算子的 SM 需求
    int be_sm = peek_sm_requirement(client_idx); // BE 当前算子的 SM 需求
    int total_sm = hp_sm + be_sm;

    if (total_sm <= config_.num_sms) return true;

    // HP 队列为空 → 无竞争
    if (g_capture_state.client_queues[0] &&
        g_capture_state.client_queues[0]->empty()) {
        return true;
    }

    return false;  // SM 冲突，阻塞 BE
}
```

---

## 5. cuCtxSetCurrent 缓存优化

当处于 ISOLATED 模式时，连续 dispatch 同一客户端的 kernel 无需重复调用 `cuCtxSetCurrent`。通过 `current_ctx_idx_` 缓存当前 context 索引：

```cpp
if (current_mode_ == ExecutionMode::ISOLATED && is_kernel_operation(op->type)) {
    int ctx_idx = (client_idx == 0) ? 0 : 1;
    if (ctx_idx != current_ctx_idx_) {
        cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
        current_ctx_idx_ = ctx_idx;
    }
}
```

对于连续的同 client kernel 序列，调用次数从 O(N) 降到 O(切换次数)。在 HP 连续执行 100 个 kernel 的场景下，节省约 1-2ms 的纯调度开销。

---

## 6. API 合规性分析

### 问题：Work Queue 隔离未配置（Driver API 限制）

- **位置**：`cuGreenCtxCreate` 调用
- **违规**：CUDA 4.6 规范指出，即使 SM 已隔离，HP/BE 仍可能因共享 work queue 而被串行化。Runtime API 的 `cudaGreenCtxCreate` 提供 `numExpectedConcurrentKernels` 参数，driver 用此提示分配独立 work queue。Driver API 的 `cuGreenCtxCreate` 无此参数，work queue 隔离无法配置
- **影响**：SM 隔离有效，但 work queue 串行化风险仍存在。实测中若 HP/BE 被分配到同一 work queue，即使 SM 空闲，BE 也需等待 HP 完全完成
- **修复选项**：
  - 短期：通过 `CUDA_DEVICE_MAX_CONNECTIONS` 环境变量增大 work queue 数量，降低碰撞概率
  - 长期：迁移到 Runtime API，使用 `cudaGreenCtxCreate(..., numExpectedConcurrentKernels=1)`
- **状态**：⚠️ 未修复，Driver API 限制

---

## 7. 参数配置

| 参数 | 默认值 | 作用 | 环境变量 |
|------|--------|------|---------|
| HP SM 数 | `56` | HP Green Context 的 SM 分区大小 | `ORION_HP_SMS` |
| BE SM 数 | `56` | BE Green Context 的 SM 分区大小 | `ORION_BE_SMS` |
| 前瞻窗口 | `3` | 向前扫描多少步检测未来 SM 冲突 | `ORION_LOOKAHEAD_WINDOW` |
| 冷却期 | `5.0 ms` | ISOLATED → DEFAULT 的最短等待时间 | `ORION_MODE_COOLDOWN_MS` |
| 切换点阈值 | `1e6 ns (1ms)` | 短于此时长的 kernel 被视为好的切换点 | 代码内常量 |
| SM 阈值 | `num_sms / 2` | 进入 ISOLATED 的 SM 总需求触发值 | `--sm-threshold` |

---

## 8. 验收标准

### 8.1 功能正确性

- 未争用时，系统保持在 DEFAULT
- 争用出现时，系统进入 ISOLATED
- 争用结束后，经冷却期 + 前瞻确认 + 好切换点后退出 ISOLATED

### 8.2 Trace 行为

- DEFAULT 阶段能看到 HP/BE 自由竞争
- ISOLATED 阶段能看到 HP/BE 在各自轨道稳定并发
- 模式切换次数有限，不会在短时间内高频震荡
- 切换倾向于发生在短 kernel（LayerNorm/Activation）处

### 8.3 指标变化

- HP 时间不能比静态 Green Context 方案更差太多
- 总时长应优于"全程串行等待"的普通 Orion
- 在轻争用区间，应优于"始终静态隔离"
- 前瞻决策的 overhead（多次 `peek_sm_requirement` 调用）应可忽略

---

## 9. 未来扩展方向

### 9.1 相位感知调度

实现方向四（3.5 节），通过监测 HP/BE 的 `seen[]` 相位差动态调整策略参数（冷却期、前瞻窗口大小）。

### 9.2 异构 SM 分区

当前 CUDA 12.8 限制为同构 56/56 分区。升级到支持异构分区的 API 后，可以根据 HP/BE 的 SM 需求比例动态调整分区比例（如 HP 密集时 72/40，BE 密集时 48/64）。

### 9.3 Runtime API 迁移

迁移到 CUDA Runtime API 的 `cudaGreenCtxCreate` 以获得 work queue 隔离能力，解决当前 Driver API 下 HP/BE 可能共享 work queue 被串行化的问题。

### 9.4 算子级别动态分区

更激进的方案：不是在 DEFAULT/ISOLATED 两种模式间切换，而是为每个算子动态计算最优 SM 分区。例如 HP 需要 80 SM、BE 需要 40 SM 时，按 80/34 分区。这需要支持运行时重新 split SM 资源，当前 CUDA API 不支持。
