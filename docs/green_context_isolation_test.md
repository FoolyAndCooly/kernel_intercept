# Green Context Isolation Test: 干扰隔离机制详解

## 1. 问题背景

### 1.1 原始对称测试为何无法展示 GC 加速

原始 Case 1/2 使用**两个完全相同的 kernel**（相同 blocks、相同 work_iters），在 H100（114 SMs）上测试。这在数学上**不可能**展示 GC 吞吐量加速，原因如下：

**H100 Block Scheduler 的串行化行为**：当两个 kernel 同时从不同 stream 提交，且总 blocks > 114 时，GPU 硬件调度器不会交错调度——它会先把一个 kernel 的全部 blocks 调度完，另一个排队等待。

以 `--launch-sms 64` 为例（每个 kernel 64 blocks）：

```
普通 streams（串行化）：

t=0ms      t=34ms      t=68ms
|---Kernel A (64 blocks on 114 SMs)---|
            |---Kernel B (64 blocks on 114 SMs)---|

  A_event_start=0    A_event_end=34ms   → 但 A_event 实测=68ms！
  B_event_start=0    B_event_end=68ms   → B_event 实测=34ms
```

为什么 A_event = 68ms？因为 `cudaEventRecord(start_A, streamA)` 在 t=0 就被 GPU 处理了（它只是一个时间戳标记），但 A 的 blocks 实际要等 B 先占完 SM 再释放。哪个 kernel "赢得"优先调度是非确定性的，所以实测中 A/B 的 68ms/34ms 会交替出现。

**关键等式**：

| 场景 | 计算公式 | Wall Time |
|------|----------|-----------|
| Baseline 串行 | `2 × ceil(N/114) × T_wave` | `2 × 34ms = 68ms` |
| GC 并行 (56/56) | `ceil(N/56) × T_wave` | `ceil(64/56) × 34ms = 2 × 34ms = 68ms` |

`ceil(N/56)` 总是 `≥ 2 × ceil(N/114)`（对 N > 56），所以 GC 永远不会更快。

### 1.2 不同 launch_sms 的实测验证

| launch_sms | Baseline Wall | GC Wall | 原因 |
|------------|---------------|---------|------|
| 56 | 34ms | 34ms | 56+56=112 ≤ 114，baseline 也重叠 |
| 64 | 68ms | 68ms | 串行 2×34 = GC 2 waves × 34 |
| 112 | 68ms | 68ms | 同上 |
| 114 | 68ms | **102ms** | GC ceil(114/56)=**3** waves，反而更慢 |

## 2. 隔离测试设计（Case 3/4/5）

### 2.1 核心思想

GC 的价值不是提升两个相同 kernel 的吞吐量，而是**在有干扰的情况下保护延迟敏感型 kernel**。

为此设计**非对称工作负载**：
- **Interferer（干扰者）**：大量 blocks + 长时间计算，模拟 GPU 上的批处理任务
- **Target（目标）**：少量 blocks + 短时间计算，模拟延迟敏感的在线推理请求

### 2.2 工作负载参数

```
Interferer:
  blocks     = sm_count (114 on H100)    // 填满整个 GPU
  work_iters = default × 4               // 4 倍计算量
  单 kernel 独占 GPU 时间 ≈ ceil(114/114) × 4 × 34ms = 136ms

Target:
  blocks     = kGreenContextSms (56)      // 恰好 1 wave on 56-SM 分区
  work_iters = default × 1               // 正常计算量
  单 kernel 独占 GPU 时间 ≈ ceil(56/114) × 34ms = 34ms
```

### 2.3 三个测试 Case

#### Case 3: Target 独立运行（基线延迟）

```
t=0          t=34ms
|---Target (56 blocks on 114 SMs, 1 wave)---|

target_event ≈ 34ms
```

Target 独占 GPU，56 blocks 在 114 SMs 上只需 1 wave，耗时 ~34ms。这是 target 的**理想延迟**。

#### Case 4: Interferer + Target 在普通 streams 上（无隔离）

两个 kernel 从两个 host 线程同时提交，通过 `LaunchGate` 原子变量确保同步释放：

```cpp
// 两个线程各自准备好后:
gate->ready.fetch_add(1);         // 报告就绪
while (!gate->go.load()) yield(); // 等待信号

// 主线程检测到两个线程都就绪后:
gate.go.store(true);              // 同时释放
```

**GPU 上的实际执行**：

```
t=0                              t=136ms         t=170ms
|===== Interferer (114 blocks × 4x work) =====|
                                  |--- Target (56 blocks × 1x work) ---|

Interferer 占满全部 114 SMs，持续 136ms
Target 的 blocks 排队等待，直到 Interferer 完成后才获得 SM

target_event = 等待时间(136ms) + 执行时间(34ms) ≈ 170ms
```

**为什么 target 必须等待？**

H100 的 block scheduler 采用"先到先服务"策略。当 Interferer 的 114 个 blocks 已经占据了全部 114 个 SM（每个 SM 因 shared memory 限制只能跑 1 block），Target 的 56 个 blocks 没有任何空闲 SM 可用，只能等 Interferer 的 blocks 逐个完成后才能上 SM。

由于 Interferer 的所有 blocks 做相同工作量（4x iterations），它们几乎同时完成。所以 Target 等待了完整的 136ms。

**target 延迟从 34ms 退化到 ~170ms，退化 5 倍。**

#### Case 5: Interferer + Target 在 Green Context 分区上（硬件隔离）

```
Green Context 分区: GC0 = 56 SMs, GC1 = 56 SMs

GC0 (Interferer):
t=0                                                    t=408ms
|=== wave1 (56 blocks) ===|=== wave2 (56 blocks) ===|== wave3 (2 blocks) ==|
       136ms                     136ms                     136ms

GC1 (Target):
t=0          t=34ms
|--- 56 blocks, 1 wave ---|

target_event ≈ 34ms  ✓ (与独立运行相同！)
```

**为什么 target 不受影响？**

Green Context 通过 `cuDevSmResourceSplitByCount` 在**硬件层面**将 114 个 SM 分成两个独立的 56-SM 分区。每个分区有自己独立的 block scheduler。Interferer 只能在 GC0 的 56 个 SM 上调度，**物理上无法触及** GC1 的 56 个 SM。

Interferer 受到惩罚：原本 1 wave（114 SMs）变成 3 waves（ceil(114/56)=3），耗时从 136ms 增加到 408ms。但 **Target 的延迟被完美保护**，始终是 34ms。

### 2.4 预期结果汇总

```
=== Isolation Test Summary ===
target alone:                 ~34 ms
target + interferer (normal): ~170 ms  (5.0x degradation)
target + interferer (GC):     ~34 ms   (1.0x degradation)
GC isolation speedup for target: ~5.00x
```

## 3. 为什么 1 block/SM 的约束至关重要

kernel 通过申请大量 dynamic shared memory 来强制 1 block/SM：

```cpp
const int requested_smem = caps.shared_mem_per_sm / 2 + 1024;
// H100: 233472 / 2 + 1024 = 117760 bytes

// 验证: 每 SM 最多 1 block
cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) == 1
```

H100 每个 SM 有 233472 bytes shared memory。每个 block 请求 117760 bytes。`233472 / 117760 = 1.98`，向下取整 = 1 block/SM。

这个约束确保：
- **blocks 数 = 所需 SM 数**：N blocks 恰好需要 N 个 SM
- **波次计算精确**：`waves = ceil(blocks / available_SMs)`
- **串行化效果最大化**：没有 time-multiplexing 来隐藏延迟

## 4. 时间线对比图

```
                0ms    34ms    68ms   102ms   136ms   170ms   ~408ms
                |       |       |       |       |       |       |
Case 3 (solo)   [Target]
                ^^^^^^^^ 34ms

Case 4 (normal) [========= Interferer (all 114 SMs) =========][Target]
                                                                ^^^^^^^ 34ms
                Target event: |<-------------- ~170ms ----------------->|

Case 5 (GC)     GC0: [==== Interferer wave1 ====][== wave2 ==][= w3 =]
                GC1: [Target]
                      ^^^^^^^^ 34ms (unaffected!)
```

## 5. 实际应用场景

此测试模拟的典型场景：

| 角色 | 真实工作负载 | 测试中的模拟 |
|------|-------------|-------------|
| Interferer | 大 batch 训练 / 离线推理 | 114 blocks, 4x work |
| Target | 在线推理请求 (latency-critical) | 56 blocks, 1x work |

**无 Green Context**：在线推理请求必须等待训练任务释放 SM，延迟不可预测。

**有 Green Context**：在线推理始终在独立的 56-SM 分区运行，延迟恒定，不受训练任务影响。代价是训练任务变慢（SM 减少），但这通常是可接受的 trade-off。

## 6. 运行方法

```bash
nvcc -std=c++17 -O3 tests/test_dual_sm_green_context.cu \
     -o test_dual_sm_green_context -lcuda -lcudart -lnvToolsExt

# 运行完整测试（Case 1-5）
./test_dual_sm_green_context --launch-sms 64

# 自定义参数
./test_dual_sm_green_context --launch-sms 64 --work-iters 2097152 --measure-rounds 10
```
