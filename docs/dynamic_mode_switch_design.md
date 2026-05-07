# 基于二维动态规划的模式切换与 SM 分配算法设计

## 1. 问题定义

### 1.1 场景

H200 GPU（132 SMs），两个服务（HP / BE）各自运行一条算子 pipeline。每个算子完成后立即提交下一个，**两个服务异步推进**。

**关键特性**：
- HP 和 BE 不同步：HP 完成 A1 后立即开始 A2，不等 BE
- 算子数量可能不同：HP 可能有 N 个算子，BE 可能有 M 个算子
- 算子时长不同：导致两个服务在不同时刻完成各自的算子

### 1.2 目标

设计一个基于**二维动态规划**的全局最优切换策略，实现：

1. **全局最优决策**：考虑整个 iteration 的所有算子，而非局部贪心
2. **异步推进处理**：正确处理两个服务的异步执行
3. **切换代价感知**：自动考虑切换对未来状态的影响
4. **动态 SM 分配**：为 ISOLATED 模式搜索最优 SM 分区

### 1.3 核心洞察

**问题本质**：这是一个**序贯决策问题**，当前决策（保持/切换）会影响未来的状态和代价。

**解决方案**：二维动态规划
- **状态**：(i, j, mode) 表示 HP 完成前 i 个算子，BE 完成前 j 个算子，当前处于 mode
- **决策**：在每个状态转移时，选择保持或切换模式
- **目标**：最小化从 (0, 0, initial_mode) 到 (N, M, *) 的总时间

### 1.4 与现有架构的关系

本模块复用现有的全部基础设施，仅替换决策逻辑：

```
现有架构（保持不变）：
├── switch_mode()           # 模式切换执行（cudaStreamSynchronize + context 切换）
├── peek_sm_requirement()   # 前瞻读取算子 SM 需求
├── peek_kernel_duration()  # 前瞻读取算子执行时长
├── seen[]                  # 程序计数器
├── op_info_vector[]        # 算子 profile 序列
├── GreenContext 资源管理    # init/destroy/stream/cublas
└── run() 主循环骨架        # 轮询 + dispatch

本模块（新增，可拔插）：
└── DPModeDecisionPolicy    # 基于二维DP的决策策略
    ├── solve_dp()           # 求解全局最优切换策略
    ├── compute_concurrent_time()  # 计算两算子并发执行时间
    ├── search_optimal_partition() # 搜索最优 SM 分区
    └── execute_plan()       # 执行规划好的切换策略
```

---

## 2. 粗状态 + Pareto 前沿标签建模

### 2.1 问题分析：为什么经典单值 DP 不够

经典 DP 用 `dp[i][j][mode]` 表示"HP 完成 i 个、BE 完成 j 个"时的最优值。但异步推进下，到达同一个 `(i, j, mode)` 时，**另一侧算子的剩余执行时间 `rest` 取决于历史路径**，不能由 `(i, j, mode)` 唯一确定。

例如到达 `(2, 1, DEFAULT)` 的两条路径：

```
路径A: (0,0) → (1,0) → (2,0) → (2,1)   →  hp_rest = f(路径A的时间差)
路径B: (0,0) → (1,0) → (1,1) → (2,1)   →  hp_rest = f(路径B的时间差)
```

两条路径的 `hp_rest` 不同，后续最优决策也不同。因此不能用单个数值 `dp[i][j][mode]` 表示。

**解决方案**：将 `rest` 从"隐含量"提升为标签的一部分，每个粗状态维护一组 **Pareto 非支配标签**，用支配关系剪枝控制规模。

### 2.1.1 粗状态定义

```
粗状态 C = (i, j, mode, side)

其中：
- i ∈ [0, N]: HP 已完成前 i 个算子
- j ∈ [0, M]: BE 已完成前 j 个算子
- mode ∈ {DEFAULT, ISOLATED}: 当前模式
- side ∈ {HP_BOUNDARY, BE_BOUNDARY}: 谁刚到达算子边界
```

**side 的物理含义**：

| side | 含义 | HP 状态 | BE 状态 |
|------|------|---------|---------|
| HP_BOUNDARY | HP 刚完成第 i 个算子 | 在边界，即将启动第 i+1 个 | 第 j+1 个正在运行，剩余 rest |
| BE_BOUNDARY | BE 刚完成第 j 个算子 | 第 i+1 个正在运行，剩余 rest | 在边界，即将启动第 j+1 个 |


**粗状态空间大小**：O(N × M × 2 × 2)

### 2.1.2 标签定义

每个粗状态 C 上维护一组标签：

```
标签 L = (t, rest)

其中：
- t: 从初始状态到达此粗状态的累计时间
- rest: 非边界侧当前算子的剩余执行时间
  - side = HP_BOUNDARY 时，rest = BE 当前算子的剩余时间
  - side = BE_BOUNDARY 时，rest = HP 当前算子的剩余时间
  - rest = 0 表示两边都在边界（特殊情况）
```

### 2.1.3 支配关系与剪枝

在同一粗状态 C 下，标签 A = (t₁, r₁) **支配** 标签 B = (t₂, r₂) 当且仅当：

```
t₁ ≤ t₂  且  r₁ ≤ r₂  （至少一个严格小于）
```

**直觉**：前缀时间更短、另一侧剩余也更短的路径，在任何后续决策下都不会更差。

每个粗状态只保留 **Pareto 前沿**（互不支配的标签集合），被支配的标签直接丢弃。

### 2.2 初始条件与终止条件

**初始条件**：

```
frontier[(0, 0, initial_mode, HP_BOUNDARY)] = { (t=0, rest=0) }
```

两个服务都未开始，处于初始模式，两边都在边界（rest=0）。

**终止条件**：

```
答案 = min { t | (t, 0) ∈ frontier[(N, M, *, *)] }
```

两个服务都完成所有算子，取所有到达终止粗状态的标签中最小的 t。

### 2.3 转移规则

从一个粗状态的标签出发，根据 side 类型分两种情况生成后继标签。当 `rest = 0` 时，两边都在边界，作为各情况内的子分支处理。

**记号约定**：
- `new_mode = ISOLATED if mode == DEFAULT else DEFAULT`
- `switch_overhead`：模式切换的固定开销（context 重建等，不含 sync 等待）

---

#### 情况 1：从 HP_BOUNDARY 状态出发

粗状态 `(i, j, mode, HP_BOUNDARY)`，标签 `(t, r)`。
HP 刚完成第 i 个算子，BE 的第 j+1 个算子还剩 r。

**1a. Keep mode，r = 0（两边都在边界），i < N 且 j < M**

两个算子同时从头并发启动：

```python
hp_time, be_time = compute_concurrent_time(hp[i], be[j], mode)

if hp_time <= be_time:
    → (i+1, j, mode, HP_BOUNDARY)  标签 (t + hp_time, be_time - hp_time)
else:
    → (i, j+1, mode, BE_BOUNDARY)  标签 (t + be_time, hp_time - be_time)
```

**1b. Keep mode，r > 0，i < N**

HP 立即启动第 i+1 个算子，与 BE 剩余部分并发：

```python
hp_full = compute_op_time(hp[i], mode, ...)

if hp_full <= r:
    → (i+1, j, mode, HP_BOUNDARY)  标签 (t + hp_full, r - hp_full)
else:
    → (i, j+1, mode, BE_BOUNDARY)  标签 (t + r, hp_full - r)
```

**1c. Keep mode，HP 已全部完成（i == N）**

```python
if r > 0:
    # 等 BE 当前算子跑完
    → (N, j+1, mode, HP_BOUNDARY)  标签 (t + r, 0)
else:
    # 两边都在边界，只有 BE 有剩余算子，独占全部 SM
    if j < M:
        be_alone = ceil(be[j].sm / total_sm) * be[j].wave_time
        → (N, j+1, mode, HP_BOUNDARY)  标签 (t + be_alone, 0)
```

**1d. Keep mode，r = 0，单侧已完成（j == M，i < N）**

```python
hp_alone = ceil(hp[i].sm / total_sm) * hp[i].wave_time
→ (i+1, M, mode, HP_BOUNDARY)  标签 (t + hp_alone, 0)
```

**1e. Switch mode**

```python
if r > 0:
    # sync BE 后切换（代价 = r + switch_overhead）
    → (i, j+1, new_mode, HP_BOUNDARY)  标签 (t + r + switch_overhead, 0)
else:
    # 两边都在边界，无需 sync（代价 = switch_overhead）
    → (i, j, new_mode, HP_BOUNDARY)  标签 (t + switch_overhead, 0)
```

---

#### 情况 2：从 BE_BOUNDARY 状态出发

与情况 1 **对称**。粗状态 `(i, j, mode, BE_BOUNDARY)`，标签 `(t, r)`。
BE 刚完成第 j 个算子，HP 的第 i+1 个算子还剩 r。

**2a. Keep mode，r = 0（两边都在边界），i < N 且 j < M**

```python
hp_time, be_time = compute_concurrent_time(hp[i], be[j], mode)

if hp_time <= be_time:
    → (i+1, j, mode, HP_BOUNDARY)  标签 (t + hp_time, be_time - hp_time)
else:
    → (i, j+1, mode, BE_BOUNDARY)  标签 (t + be_time, hp_time - be_time)
```

**2b. Keep mode，r > 0，j < M**

```python
be_full = compute_op_time(be[j], mode, ...)

if be_full <= r:
    → (i, j+1, mode, BE_BOUNDARY)  标签 (t + be_full, r - be_full)
else:
    → (i+1, j, mode, HP_BOUNDARY)  标签 (t + r, be_full - r)
```

**2c. Keep mode，BE 已全部完成（j == M）**

```python
if r > 0:
    → (i+1, M, mode, BE_BOUNDARY)  标签 (t + r, 0)
else:
    if i < N:
        hp_alone = ceil(hp[i].sm / total_sm) * hp[i].wave_time
        → (i+1, M, mode, BE_BOUNDARY)  标签 (t + hp_alone, 0)
```

**2d. Keep mode，r = 0，单侧已完成（i == N，j < M）**

```python
be_alone = ceil(be[j].sm / total_sm) * be[j].wave_time
→ (N, j+1, mode, BE_BOUNDARY)  标签 (t + be_alone, 0)
```

**2e. Switch mode**

```python
if r > 0:
    → (i+1, j, new_mode, BE_BOUNDARY)  标签 (t + r + switch_overhead, 0)
else:
    → (i, j, new_mode, BE_BOUNDARY)  标签 (t + switch_overhead, 0)
```

### 2.4 关键函数

#### 2.4.1 compute_concurrent_time（rest = 0 时使用）

两个算子**同时从头**并发执行时，各自的完成时间。仅在 rest = 0（两边都在边界）时使用。

**ISOLATED 模式**（两个算子完全独立）：

```python
def compute_concurrent_time_isolated(hp_op, be_op, s1, s2):
    hp_time = math.ceil(hp_op.sm / s1) * hp_op.wave_time
    be_time = math.ceil(be_op.sm / s2) * be_op.wave_time
    return hp_time, be_time
```

**DEFAULT 模式**（共享 SM，HP 优先调度）：

```python
def compute_concurrent_time_default(hp_op, be_op, total_sm):
    sm_a, sm_b = hp_op.sm, be_op.sm
    t_a, t_b = hp_op.wave_time, be_op.wave_time
    
    if sm_a + sm_b <= total_sm:
        return t_a, t_b
    else:
        k = math.ceil((sm_a + sm_b - total_sm) / total_sm)
        hp_time = (k + 1) * t_a
        be_time = k * t_a + t_b
        return hp_time, be_time
```

#### 2.4.2 compute_op_time（rest > 0 时使用）

一侧启动新算子，另一侧正在运行（剩余 rest > 0）。返回新算子的**完整执行时间**。

**ISOLATED 模式**（各自独立，不受另一侧影响）：

```python
def compute_op_time_isolated(op, s_allocated):
    return math.ceil(op.sm / s_allocated) * op.wave_time
```

**DEFAULT 模式**（共享 SM，需考虑与另一侧的竞争）：

```python
def compute_op_time_default(op, total_sm):
    # 简化模型：新算子按独占全部 SM 计算
    # 理由：另一侧正在运行的算子剩余时间有限，
    #       竞争窗口较短，对新算子总时间影响较小
    return math.ceil(op.sm / total_sm) * op.wave_time
```

> **注**：DEFAULT 模式下更精确的做法是分段计算——在 rest 时间内按竞争模型、rest 之后按独占模型。
> 但这会使转移逻辑复杂化，且实际误差较小（竞争窗口 ≤ rest，通常远小于新算子总时间）。
> 后续可根据实测精度决定是否升级为分段模型。

### 2.5 切换代价

切换代价取决于 rest：

| rest | sync 等待 | 切换开销 | 总代价 |
|------|-----------|----------|--------|
| 0（两边都在边界） | 0 | switch_overhead | switch_overhead |
| > 0 | rest（等另一侧完成） | switch_overhead | rest + switch_overhead |

**物理含义**：`switch_mode()` 需要 `cudaStreamSynchronize` 等待所有正在运行的 kernel 完成，然后执行 context 切换。

- `rest`：等待正在运行的算子完成的时间（精确值，非近似）
- `switch_overhead`：context 切换的固定开销（销毁/创建 Green Context 等，约 2-5ms）

---

## 3. 求解算法：前向标签传播

### 3.1 算法概述

不再使用后向 DP 递推，而是**前向 label-setting 最短路**：

1. 从初始粗状态出发，按累计时间 t 从小到大处理标签
2. 每个标签生成后继标签（按 §2.3 转移规则）
3. 后继标签若不被目标粗状态的已有前沿支配，则加入前沿和优先队列
4. 答案：终止粗状态中最小的 t

**与经典 Dijkstra 的区别**：每个粗状态不是"一个最优值"，而是"一组 Pareto 非支配标签"。这是多目标最短路（bi-criteria shortest path）的标准做法。

### 3.2 完整算法

```python
import heapq
from collections import defaultdict

HP_BOUNDARY, BE_BOUNDARY = 0, 1
DEFAULT, ISOLATED = 0, 1

def solve_label_setting(hp_ops, be_ops, initial_mode, total_sm, partition):
    """
    前向标签传播求解最优切换策略
    
    参数：
        hp_ops: HP 算子序列 [{sm, wave_time}, ...]
        be_ops: BE 算子序列 [{sm, wave_time}, ...]
        initial_mode: 初始模式
        total_sm: GPU 总 SM 数
        partition: ISOLATED 模式的 SM 分配 (s1, s2)
    
    返回：
        min_time: 最小总时间
        plan: 最优切换计划（切换点列表）
    """
    N, M = len(hp_ops), len(be_ops)
    s1, s2 = partition
    
    # frontier[coarse_state] = [(t, rest)] 的 Pareto 前沿
    frontier = defaultdict(list)
    
    # parent[coarse_state][(t, rest)] = (prev_state, prev_label, action)
    parent = defaultdict(dict)
    
    # 优先队列：(t, rest, i, j, mode, side)
    pq = []
    
    # 初始标签：两边都在边界（rest=0）
    init_state = (0, 0, initial_mode, HP_BOUNDARY)
    init_label = (0.0, 0.0)
    frontier[init_state].append(init_label)
    heapq.heappush(pq, (0.0, 0.0, 0, 0, initial_mode, HP_BOUNDARY))
    
    def add_label(state, label, prev_state, prev_label, action):
        """尝试将标签加入前沿，若不被支配则加入"""
        t, r = label
        front = frontier[state]
        
        for (ft, fr) in front:
            if ft <= t and fr <= r:
                return  # 被支配，丢弃
        
        frontier[state] = [(ft, fr) for (ft, fr) in front
                           if not (t <= ft and r <= fr)]
        frontier[state].append(label)
        parent[state][label] = (prev_state, prev_label, action)
        
        i, j, mode, side = state
        heapq.heappush(pq, (t, r, i, j, mode, side))
    
    # ---- 主循环 ----
    while pq:
        t, r, i, j, mode, side = heapq.heappop(pq)
        label = (t, r)
        state = (i, j, mode, side)
        
        if label not in frontier[state]:
            continue
        
        if i == N and j == M:
            plan = _backtrack(parent, state, label)
            return t, plan
        
        new_mode = ISOLATED if mode == DEFAULT else DEFAULT
        
        if side == HP_BOUNDARY:
            if r == 0:
                # 两边都在边界
                if i < N and j < M:
                    # 1a. Keep: 两个算子同时启动
                    hp_t, be_t = compute_concurrent_time(
                        hp_ops[i], be_ops[j], mode, total_sm, s1, s2)
                    if hp_t <= be_t:
                        add_label((i+1, j, mode, HP_BOUNDARY),
                                  (t + hp_t, be_t - hp_t),
                                  state, label, 'keep')
                    else:
                        add_label((i, j+1, mode, BE_BOUNDARY),
                                  (t + be_t, hp_t - be_t),
                                  state, label, 'keep')
                elif i == N and j < M:
                    # 1c. 只有 BE
                    be_alone = ceil(be_ops[j].sm / total_sm) * be_ops[j].wave_time
                    add_label((N, j+1, mode, HP_BOUNDARY),
                              (t + be_alone, 0.0),
                              state, label, 'keep')
                elif j == M and i < N:
                    # 1d. 只有 HP
                    hp_alone = ceil(hp_ops[i].sm / total_sm) * hp_ops[i].wave_time
                    add_label((i+1, M, mode, HP_BOUNDARY),
                              (t + hp_alone, 0.0),
                              state, label, 'keep')
                
                # 1e. Switch: 无需 sync
                add_label((i, j, new_mode, HP_BOUNDARY),
                          (t + SWITCH_OVERHEAD, 0.0),
                          state, label, 'switch')
            else:
                # r > 0: BE 正在运行
                if i < N:
                    # 1b. Keep: HP 启动下一个算子
                    hp_full = compute_op_time(hp_ops[i], mode, total_sm, s1)
                    if hp_full <= r:
                        add_label((i+1, j, mode, HP_BOUNDARY),
                                  (t + hp_full, r - hp_full),
                                  state, label, 'keep')
                    else:
                        add_label((i, j+1, mode, BE_BOUNDARY),
                                  (t + r, hp_full - r),
                                  state, label, 'keep')
                else:
                    # HP 已全部完成，等 BE 跑完
                    add_label((N, j+1, mode, HP_BOUNDARY),
                              (t + r, 0.0),
                              state, label, 'keep')
                
                # Switch: sync BE 后切换
                add_label((i, j+1, new_mode, HP_BOUNDARY),
                          (t + r + SWITCH_OVERHEAD, 0.0),
                          state, label, 'switch')
        
        elif side == BE_BOUNDARY:
            if r == 0:
                # 两边都在边界（对称）
                if i < N and j < M:
                    hp_t, be_t = compute_concurrent_time(
                        hp_ops[i], be_ops[j], mode, total_sm, s1, s2)
                    if hp_t <= be_t:
                        add_label((i+1, j, mode, HP_BOUNDARY),
                                  (t + hp_t, be_t - hp_t),
                                  state, label, 'keep')
                    else:
                        add_label((i, j+1, mode, BE_BOUNDARY),
                                  (t + be_t, hp_t - be_t),
                                  state, label, 'keep')
                elif j == M and i < N:
                    hp_alone = ceil(hp_ops[i].sm / total_sm) * hp_ops[i].wave_time
                    add_label((i+1, M, mode, BE_BOUNDARY),
                              (t + hp_alone, 0.0),
                              state, label, 'keep')
                elif i == N and j < M:
                    be_alone = ceil(be_ops[j].sm / total_sm) * be_ops[j].wave_time
                    add_label((N, j+1, mode, BE_BOUNDARY),
                              (t + be_alone, 0.0),
                              state, label, 'keep')
                
                add_label((i, j, new_mode, BE_BOUNDARY),
                          (t + SWITCH_OVERHEAD, 0.0),
                          state, label, 'switch')
            else:
                # r > 0: HP 正在运行
                if j < M:
                    be_full = compute_op_time(be_ops[j], mode, total_sm, s2)
                    if be_full <= r:
                        add_label((i, j+1, mode, BE_BOUNDARY),
                                  (t + be_full, r - be_full),
                                  state, label, 'keep')
                    else:
                        add_label((i+1, j, mode, HP_BOUNDARY),
                                  (t + r, be_full - r),
                                  state, label, 'keep')
                else:
                    add_label((i+1, M, mode, BE_BOUNDARY),
                              (t + r, 0.0),
                              state, label, 'keep')
                
                add_label((i+1, j, new_mode, BE_BOUNDARY),
                          (t + r + SWITCH_OVERHEAD, 0.0),
                          state, label, 'switch')
    
    return float('inf'), []


def _backtrack(parent, end_state, end_label):
    """回溯最优路径，提取切换点"""
    plan = []
    state, label = end_state, end_label
    
    while state in parent and label in parent[state]:
        prev_state, prev_label, action = parent[state][label]
        if action == 'switch':
            i, j, mode, side = prev_state
            _, _, to_mode, _ = state
            plan.append({
                'hp_idx': i, 'be_idx': j,
                'from_mode': mode, 'to_mode': to_mode
            })
        state, label = prev_state, prev_label
    
    plan.reverse()
    return plan
```

### 3.3 复杂度分析

- **粗状态数**：O(N × M × 2 × 2) = O(N × M)
- **每个粗状态的非支配标签数**：设为 K
  - K 受 rest 值域约束（rest ≤ 最大算子时长）
  - 实际中 K 很小（通常 1-5），因为路径趋同、大量标签被支配
- **总标签数**：O(N × M × K)
- **每个标签的处理**：O(K)（支配检查）+ O(log(NMK))（堆操作）
- **总时间复杂度**：O(N × M × K² × log(NMK))
- **空间复杂度**：O(N × M × K)

**实际数值**（N = M = 100, K ≈ 3）：
- 总标签数 ≈ 100 × 100 × 2 × 3 × 3 = 180,000
- 每个标签处理 < 1μs
- 总计算时间 < 200ms（保守估计）
- 在 iteration 开始时一次性计算，开销可接受

### 3.4 SM 分区搜索

与原方案相同，通过外层枚举 ISOLATED 模式的 SM 分区找全局最优：

```python
def find_optimal_plan(hp_ops, be_ops, initial_mode, total_sm):
    best_time = float('inf')
    best_plan = None
    best_partition = None
    
    for s1 in range(8, total_sm - 7, 8):
        s2 = total_sm - s1
        
        time, plan = solve_label_setting(
            hp_ops, be_ops, initial_mode, total_sm, (s1, s2))
        
        if time < best_time:
            best_time = time
            best_plan = plan
            best_partition = (s1, s2)
    
    return best_time, best_plan, best_partition
```

**总复杂度**：O(S/8 × N × M × K²)


## 4. 运行时执行

### 4.1 决策时机

**在每个 iteration 开始时调用一次 DP 求解**，生成整个 iteration 的切换计划。

**Iteration 边界检测**：

```cpp
bool Scheduler::is_iteration_boundary() {
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);
    
    // 检查两个服务是否都完成了一轮
    for (int i = 0; i < num_clients_; i++) {
        if (g_orion_state.seen[i] == 0) {
            return false;  // 还未开始
        }
        
        int n = g_orion_state.op_info_vector[i].size();
        if (n > 0 && g_orion_state.seen[i] % n != 0) {
            return false;  // 未完成一轮
        }
    }
    
    return true;
}
```

### 4.2 执行计划数据结构

#### 4.2.1 切换点（SwitchPoint）

切换点表示在哪个状态切换模式。状态 (i, j) 表示 HP 完成了前 i 个算子，BE 完成了前 j 个算子。

```cpp
struct SwitchPoint {
    int hp_idx;           // HP 完成第 hp_idx 个算子后
    int be_idx;           // BE 完成第 be_idx 个算子后
    ExecutionMode to_mode;  // 切换到的目标模式
};
```

**语义**：
- `hp_idx = 5, be_idx = 4, to_mode = ISOLATED` 表示：
  - 当 HP 完成了至少 5 个算子 **且** BE 完成了至少 4 个算子时
  - 切换到 ISOLATED 模式

**为什么是"至少"？**

因为两个服务异步推进，可能出现：
- HP 先完成第 5 个算子（此时 BE 可能才完成 3 个）
- 稍后 BE 完成第 4 个算子
- 此时两个条件都满足，执行切换

#### 4.2.2 迭代计划（IterationPlan）

```cpp
struct IterationPlan {
    std::vector<SwitchPoint> switches;  // 切换点列表（按执行顺序排序）
    int hp_sm_count;   // ISOLATED 模式的 HP SM 数
    int be_sm_count;   // ISOLATED 模式的 BE SM 数
    float expected_time;  // 预期总时间（ms）
};
```

**示例**：

```cpp
IterationPlan plan = {
    .switches = {
        {5, 4, ExecutionMode::ISOLATED},  // 第一个切换点
        {8, 9, ExecutionMode::DEFAULT}    // 第二个切换点
    },
    .hp_sm_count = 72,
    .be_sm_count = 60,
    .expected_time = 420.5
};
```

**JSON 存储格式**：

```json
{
  "switches": [
    {"hp_idx": 5, "be_idx": 4, "to_mode": "ISOLATED"},
    {"hp_idx": 8, "be_idx": 9, "to_mode": "DEFAULT"}
  ],
  "partition": [72, 60],
  "expected_time_ms": 420.5
}
```

### 4.3 运行时切换执行

#### 4.3.1 进度跟踪

在主循环中跟踪两个服务各自完成的 kernel 算子数量：

```cpp
// scheduler.h 新增成员
class Scheduler {
private:
    IterationPlan current_plan_;
    int hp_completed_ = 0;  // 当前 iteration 已完成的 HP kernel 数
    int be_completed_ = 0;  // 当前 iteration 已完成的 BE kernel 数
    
    // 已执行的切换点索引（避免重复切换）
    size_t next_switch_idx_ = 0;
};
```

#### 4.3.2 主循环集成

```cpp
void Scheduler::run() {
    while (running_.load()) {
        // ① 检测 iteration 边界
        if (is_iteration_boundary()) {
            // 加载或计算切换计划
            load_or_compute_plan();
            
            // 重置计数器
            hp_completed_ = 0;
            be_completed_ = 0;
            next_switch_idx_ = 0;
        }
        
        // ② 轮询所有客户端队列
        for (int i = 0; i < num_clients_; i++) {
            auto op = g_capture_state.client_queues[i]->peek();
            if (!op) continue;
            
            // BE 调度判断
            if (i > 0 && !orion_should_schedule(op, i)) continue;
            
            g_capture_state.client_queues[i]->try_pop();
            
            // ③ 执行算子
            cudaStream_t stream = select_stream(i);
            cublasHandle_t handle = select_cublas_handle(i);
            execute_operation(op, stream, handle);
            
            // ④ 更新进度并检查切换点
            if (is_kernel_operation(op->type)) {
                if (i == 0) {
                    hp_completed_++;
                } else {
                    be_completed_++;
                }
                
                // 检查是否到达切换点
                check_and_execute_switch();
            }
        }
    }
}
```

#### 4.3.3 切换点检查与执行

**方法 1：顺序检查（推荐）**

假设切换点列表已按执行顺序排序，只需检查下一个切换点：

```cpp
void Scheduler::check_and_execute_switch() {
    // 如果所有切换点都已执行，直接返回
    if (next_switch_idx_ >= current_plan_.switches.size()) {
        return;
    }
    
    const auto& sp = current_plan_.switches[next_switch_idx_];
    
    // 检查是否两个服务都到达了切换点
    if (hp_completed_ >= sp.hp_idx && be_completed_ >= sp.be_idx) {
        LOG_INFO("[DP-SWITCH] Switching at (%d, %d) to %s (actual: %d, %d)",
                 sp.hp_idx, sp.be_idx,
                 sp.to_mode == ExecutionMode::ISOLATED ? "ISOLATED" : "DEFAULT",
                 hp_completed_, be_completed_);
        
        // 执行切换
        switch_mode(sp.to_mode);
        
        // 标记为已执行
        next_switch_idx_++;
        
        // 递归检查下一个切换点（可能连续多个切换点同时满足）
        check_and_execute_switch();
    }
}
```

**方法 2：全量检查**

如果切换点列表未排序，或需要处理复杂情况：

```cpp
void Scheduler::check_and_execute_switch() {
    // 遍历所有未执行的切换点
    for (auto it = current_plan_.switches.begin(); 
         it != current_plan_.switches.end(); ) {
        
        // 检查是否两个服务都到达了切换点
        if (hp_completed_ >= it->hp_idx && be_completed_ >= it->be_idx) {
            LOG_INFO("[DP-SWITCH] Switching at (%d, %d) to %s (actual: %d, %d)",
                     it->hp_idx, it->be_idx,
                     it->to_mode == ExecutionMode::ISOLATED ? "ISOLATED" : "DEFAULT",
                     hp_completed_, be_completed_);
            
            switch_mode(it->to_mode);
            
            // 从列表中移除已执行的切换点
            it = current_plan_.switches.erase(it);
        } else {
            ++it;
        }
    }
}
```

#### 4.3.4 切换时机的精确性

**问题**：切换点 (5, 4) 表示"HP 完成 5 个且 BE 完成 4 个"，但在异步执行中，这两个条件可能在不同时刻满足。

**时间线示例**：

```
t=0:    A0, B0 开始
t=30:   A0 完成 → hp_completed = 1
t=40:   B0 完成 → be_completed = 1
...
t=150:  A4 完成 → hp_completed = 5
        检查切换点 (5, 4)：hp_completed=5 ✓, be_completed=3 ✗
        不满足，继续
t=160:  B3 完成 → be_completed = 4
        检查切换点 (5, 4)：hp_completed=5 ✓, be_completed=4 ✓
        满足！执行切换
```

**关键点**：
- 切换在"两个条件都满足"的时刻执行
- 这个时刻可能是 HP 完成某个算子后，也可能是 BE 完成某个算子后
- 使用 `>=` 而不是 `==` 判断，确保不会错过切换点

#### 4.3.5 切换点的排序

为了提高效率，预计算时应将切换点按"预期执行顺序"排序：

```python
def sort_switch_points(switches):
    """
    按预期执行顺序排序切换点
    
    排序规则：按 (hp_idx + be_idx) 升序
    """
    return sorted(switches, key=lambda sp: sp["hp_idx"] + sp["be_idx"])
```

**为什么这样排序？**

状态 (i, j) 的"执行顺序"大致对应 i + j 的值。虽然不完全精确（因为异步），但作为启发式规则足够好。

#### 4.3.6 完整示例

**预计算结果**：

```json
{
  "switches": [
    {"hp_idx": 3, "be_idx": 3, "to_mode": "ISOLATED"},
    {"hp_idx": 7, "be_idx": 7, "to_mode": "DEFAULT"}
  ],
  "partition": [72, 60]
}
```

**运行时执行**：

```
t=0:    加载计划，hp_completed=0, be_completed=0, next_switch_idx=0
        当前模式：DEFAULT

t=10:   A0 完成 → hp_completed=1
        检查切换点 (3, 3)：1 >= 3? ✗

t=20:   B0 完成 → be_completed=1
        检查切换点 (3, 3)：1 >= 3? ✗

...

t=90:   A2 完成 → hp_completed=3
        检查切换点 (3, 3)：3 >= 3 ✓, 2 >= 3? ✗

t=100:  B2 完成 → be_completed=3
        检查切换点 (3, 3)：3 >= 3 ✓, 3 >= 3 ✓
        执行切换到 ISOLATED！
        next_switch_idx=1

t=110:  A3 开始（在 ISOLATED 模式下，使用 72 个 SM）
        B3 开始（在 ISOLATED 模式下，使用 60 个 SM）

...

t=210:  A6 完成 → hp_completed=7
        检查切换点 (7, 7)：7 >= 7 ✓, 6 >= 7? ✗

t=220:  B6 完成 → be_completed=7
        检查切换点 (7, 7)：7 >= 7 ✓, 7 >= 7 ✓
        执行切换到 DEFAULT！
        next_switch_idx=2（超出范围，后续不再检查）

t=230:  A7 开始（在 DEFAULT 模式下，共享全部 132 个 SM）
        B7 开始
```

#### 4.3.7 边界情况处理

**情况 1：切换点超出算子数量**

如果 DP 计算的切换点 (10, 8)，但实际 HP 只有 9 个算子：

```cpp
// 在 check_and_execute_switch() 中
if (hp_completed_ >= sp.hp_idx && be_completed_ >= sp.be_idx) {
    // 正常执行切换
}
// 如果 hp_completed 永远达不到 sp.hp_idx，切换点永远不会触发
// 这是安全的，因为 iteration 结束后会重新规划
```

**情况 2：多个切换点同时满足**

如果由于某种原因，两个切换点同时满足（如 (3, 3) 和 (3, 4)）：

```cpp
// 方法 1 的递归调用会依次执行所有满足条件的切换点
void Scheduler::check_and_execute_switch() {
    if (next_switch_idx_ >= current_plan_.switches.size()) return;
    
    const auto& sp = current_plan_.switches[next_switch_idx_];
    if (hp_completed_ >= sp.hp_idx && be_completed_ >= sp.be_idx) {
        switch_mode(sp.to_mode);
        next_switch_idx_++;
        check_and_execute_switch();  // 递归检查下一个
    }
}
```

**情况 3：iteration 中途参数变化**

如果 batch_size 在 iteration 中途变化（不太可能，但需要考虑）：

```cpp
// 在检测到参数变化时，重新规划
if (batch_size_changed()) {
    LOG_WARN("Batch size changed mid-iteration, replanning...");
    load_or_compute_plan();
    hp_completed_ = 0;
    be_completed_ = 0;
    next_switch_idx_ = 0;
}
```

```cpp
// scheduler.h 新增成员
class Scheduler {
private:
    std::unique_ptr<DPModeDecisionPolicy> dp_policy_;
    IterationPlan current_plan_;
    int hp_completed_ = 0;  // 当前 iteration 已完成的 HP 算子数
    int be_completed_ = 0;  // 当前 iteration 已完成的 BE 算子数
};

// scheduler.cpp run() 中
void Scheduler::run() {
    while (running_.load()) {
        // ① 检测 iteration 边界
        if (is_iteration_boundary()) {
            LOG_INFO("[DP-PLAN] Iteration boundary detected, planning...");
            
            // 调用 DP 求解
            current_plan_ = dp_policy_->plan_for_iteration(
                g_orion_state.op_info_vector[0],
                g_orion_state.op_info_vector[1],
                current_mode_,
                config_.num_sms
            );
            
            LOG_INFO("[DP-PLAN] Plan generated: %zu switches, expected time %.2fms",
                     current_plan_.switches.size(),
                     current_plan_.expected_time);
            
            hp_completed_ = 0;
            be_completed_ = 0;
        }
        
        // ② 轮询所有客户端队列
        for (int i = 0; i < num_clients_; i++) {
            auto op = g_capture_state.client_queues[i]->peek();
            if (!op) continue;
            
            // BE 调度判断（现有逻辑）
            if (i > 0 && !orion_should_schedule(op, i)) continue;
            
            g_capture_state.client_queues[i]->try_pop();
            
            // ③ 执行算子
            cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i-1];
            execute_operation(op, stream, ...);
            
            // ④ 更新完成计数
            if (is_kernel_operation(op->type)) {
                if (i == 0) {
                    hp_completed_++;
                } else {
                    be_completed_++;
                }
                
                // ⑤ 检查是否到达切换点
                check_and_execute_switch(hp_completed_, be_completed_);
            }
        }
    }
}

void Scheduler::check_and_execute_switch(int hp_idx, int be_idx) {
    for (const auto& sp : current_plan_.switches) {
        if (sp.hp_idx == hp_idx && sp.be_idx == be_idx) {
            LOG_INFO("[DP-SWITCH] Switching at (%d, %d) to %s",
                     hp_idx, be_idx,
                     sp.to_mode == ISOLATED ? "ISOLATED" : "DEFAULT");
            
            switch_mode(sp.to_mode);
            break;
        }
    }
}
```

### 4.4 与现有架构的兼容性

**完全兼容**：
- 复用现有的 `switch_mode()`
- 复用现有的 `peek_sm_requirement()` 和 `peek_kernel_duration()`
- 复用现有的 `seen[]` 和 `op_info_vector[]`
- 复用现有的 Green Context 资源管理

**唯一变化**：
- 将 `decide_mode()` 的调用从"每次循环"改为"iteration 边界"
- 决策逻辑从"实时评估"改为"离线规划 + 在线执行"

---

---
