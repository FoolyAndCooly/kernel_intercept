# SM 占用预测实现现状分析

## 结论

**当前代码（scheduler.cpp）并未实现基于 SM 占用的调度决策，文档中的 `get_operator_sm_requirement()` 只是设计方案，尚未实现。**

---

## 证据 1：当前调度逻辑不使用 SM 数据

### 代码位置：`src/scheduler.cpp:494-516`

```cpp
bool Scheduler::orion_should_schedule(OperationPtr op, int client_idx) {
    (void)client_idx;  // 暂时不使用

    // 内存操作直接允许
    if (is_memory_operation(op->type)) {
        return true;
    }

    // 非 kernel 操作直接允许（如同步操作）
    if (!is_kernel_operation(op->type)) {
        return true;
    }

    // 检查 HP 队列是否为空
    // 如果 HP 队列为空，说明 HP 当前没有待执行的操作，BE 可以执行
    if (g_capture_state.client_queues[0] &&
        g_capture_state.client_queues[0]->empty()) {
        return true;
    }

    // HP 队列不为空，BE 需要等待
    return false;
}
```

**分析**：
- ❌ 没有调用 `get_operator_sm_requirement()`
- ❌ 没有检查 SM 占用情况
- ❌ 只是简单判断"HP 队列是否为空"
- ✅ 这是最简单的优先级调度，不涉及 SM 资源管理

---

## 证据 2：kernel_info.csv 的 SM 数据已加载但未使用

### 代码位置：`src/scheduler.cpp:81-150`

```cpp
static int populate_kernel_info(int client_idx, const std::string& file_path) {
    // ... 省略文件读取代码 ...

    // 解析 CSV: Name,Profile,Memory_footprint,SM_usage,Duration
    KernelProfileInfo info;
    info.name = name;
    info.profile = std::stoi(profile_str);
    info.mem = std::stoi(mem_str);
    info.sm_used = std::stoi(sm_str);      // ← SM 数据已解析
    info.duration = std::stof(dur_str);

    op_info.push_back(info);

    // 存储到全局状态
    g_orion_state.op_info_vector[client_idx] = std::move(op_info);
}
```

**分析**：
- ✅ `sm_used` 字段已从 CSV 加载
- ✅ 数据存储在 `g_orion_state.op_info_vector[client_idx][kernel_idx].sm_used`
- ❌ 但调度器从未读取这个字段
- ❌ 没有任何代码使用 `sm_used` 进行调度决策

---

## 证据 3：文档中的 `get_operator_sm_requirement()` 不存在

### 文档位置：`docs/OPERATOR_LEVEL_SM_ALLOCATION_DETAILED_DESIGN.md:1038-1049`

```cpp
int Scheduler::get_operator_sm_requirement(OperationPtr op, int client_idx) {
    // 从 kernel_info.csv 加载的 profile 数据中查询
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);

    int seen_idx = g_orion_state.seen[client_idx];
    if (seen_idx < g_orion_state.op_info_vector[client_idx].size()) {
        return g_orion_state.op_info_vector[client_idx][seen_idx].sm_used;
    }

    // 默认值：假设中等 SM 需求
    return 32;
}
```

**验证**：
```bash
# 搜索这个函数
grep -r "get_operator_sm_requirement" src/
# 结果：无匹配
```

**分析**：
- ❌ 这个函数在代码中不存在
- ❌ 只是文档中的设计方案
- ❌ 尚未实现

---

## 证据 4：`seen` 索引机制存在但未用于 SM 查询

### 代码位置：`include/scheduler.h:62-77`

```cpp
struct OrionSchedulingState {
    // Kernel Profile 信息
    std::vector<std::vector<KernelProfileInfo>> op_info_vector;

    // 每个客户端当前已调度的 kernel 数
    std::vector<int> seen;  // ← 索引存在

    // 每个客户端每次迭代的 kernel 数
    std::vector<int> num_client_kernels;

    // SM 阈值
    int sm_threshold = 0;
};
```

**分析**：
- ✅ `seen` 数组用于跟踪每个客户端执行到第几个 kernel
- ✅ 可以用来索引 `op_info_vector[client_idx][seen[client_idx]].sm_used`
- ❌ 但当前代码没有这样做
- ❌ `seen` 只在 Orion 原始论文的实现中使用，当前简化版未启用

---

## 当前实现的 SM 预测方法：无

**总结**：当前代码**完全没有实现 SM 占用预测**，调度决策基于：
1. 操作类型（内存操作 vs kernel 操作）
2. HP 队列是否为空

---

## 如何实现 SM 预测（3 种方法）

### 方法 1：离线 Profiling + 查表（推荐，数据已准备好）

**实现步骤**：
1. 数据已准备：`kernel_info.csv` 中的 `SM_usage` 列
2. 添加函数：
```cpp
int Scheduler::get_operator_sm_requirement(OperationPtr op, int client_idx) {
    std::lock_guard<std::mutex> lock(g_orion_state.mutex);

    int seen_idx = g_orion_state.seen[client_idx];
    if (seen_idx < g_orion_state.op_info_vector[client_idx].size()) {
        int sm_used = g_orion_state.op_info_vector[client_idx][seen_idx].sm_used;
        g_orion_state.seen[client_idx]++;  // 移动到下一个 kernel
        return sm_used;
    }

    return 32;  // 默认值
}
```

3. 在调度器中使用：
```cpp
bool Scheduler::orion_should_schedule(OperationPtr op, int client_idx) {
    if (!is_kernel_operation(op->type)) {
        return true;
    }

    // 获取 SM 需求
    int sm_req = get_operator_sm_requirement(op, client_idx);

    // 检查是否会冲突
    int current_sm = get_current_sm_usage();  // 需要实现
    if (current_sm + sm_req > 114) {
        return false;  // 会冲突，等待
    }

    return true;
}
```

**优势**：
- ✅ 数据已准备好（kernel_info.csv）
- ✅ 准确度高（基于真实 profiling）
- ✅ 实现简单（查表）

**局限**：
- ❌ 需要预先 profiling
- ❌ 假设每次推理的 kernel 序列固定
- ❌ 新模型需要重新 profiling

---

### 方法 2：运行时动态估算（无需 profiling）

**从 kernel launch 参数估算**：

```cpp
int estimate_sm_from_launch(dim3 gridDim, dim3 blockDim) {
    // H100 硬件参数
    const int MAX_THREADS_PER_SM = 2048;
    const int MAX_BLOCKS_PER_SM = 32;

    int total_blocks = gridDim.x * gridDim.y * gridDim.z;
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

    // 从线程数约束估算
    int blocks_per_sm = MAX_THREADS_PER_SM / max(threads_per_block, 1);
    blocks_per_sm = min(blocks_per_sm, MAX_BLOCKS_PER_SM);

    // 需要多少 SM
    int sm_needed = (total_blocks + blocks_per_sm - 1) / blocks_per_sm;
    return min(sm_needed, 114);
}
```

**在拦截层获取参数**：
```cpp
// 在 cudaLaunchKernel 拦截点
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, ...) {
    // 估算 SM 需求
    int sm_req = estimate_sm_from_launch(gridDim, blockDim);

    // 存储到 Operation 中
    op->sm_requirement = sm_req;

    // 提交到队列
    submit_to_scheduler(op);
}
```

**优势**：
- ✅ 无需 profiling
- ✅ 适用于任意模型
- ✅ 实时估算

**局限**：
- ❌ 准确度中等（缺少寄存器/共享内存信息）
- ❌ 需要修改拦截层代码

---

### 方法 3：基于算子类型的启发式规则

**GEMM 算子的 SM 估算**：

```cpp
int estimate_gemm_sm(int M, int N, int K) {
    long long flops = 2LL * M * N * K;

    if (flops > 1e11) return 114;   // 超大矩阵，占满
    if (flops > 1e10) return 64;    // 大矩阵
    if (flops > 1e9)  return 32;    // 中等矩阵
    return 8;                        // 小矩阵
}
```

**在 cuBLAS 拦截点使用**：
```cpp
cublasStatus_t cublasSgemm(..., int m, int n, int k, ...) {
    int sm_req = estimate_gemm_sm(m, n, k);
    op->sm_requirement = sm_req;
    submit_to_scheduler(op);
}
```

**优势**：
- ✅ 实现极简
- ✅ 无需 profiling
- ✅ 对 GEMM 类算子效果好

**局限**：
- ❌ 准确度低
- ❌ 只适用于特定算子类型
- ❌ 需要为每种算子写规则

---

## 推荐实施方案

**阶段 1（1 周）**：实现方法 1（离线 profiling + 查表）
- 数据已准备好，实现简单
- 准确度高，适合验证方案可行性

**阶段 2（2 周）**：添加方法 2（运行时估算）作为 fallback
- 当 profiling 数据缺失时使用
- 提高方案的通用性

**阶段 3（可选）**：优化方法 2 的准确度
- 使用 CUDA Occupancy Calculator API
- 考虑寄存器和共享内存约束

---

## kernel_info.csv 数据示例

```csv
Name,Profile,Memory_footprint,SM_usage,Duration
ampere_sgemm_32x128_tn,1,0,32,395202.0
ampere_sgemm_128x128_tn,1,0,64,507330.0
ampere_sgemm_128x128_nn,1,0,64,2644042.0
```

**字段说明**：
- `SM_usage`：该 kernel 需要的 SM 数量（从 NCU profiling 计算得出）
- `Profile`：1=计算密集型，0=访存密集型
- `Duration`：执行时间（纳秒）

**数据来源**：`profiling/get_num_blocks.py` 脚本计算
- 输入：NCU profiling 的 Grid/Block/寄存器/共享内存数据
- 输出：每个 kernel 需要的 SM 数量

---

## 总结

1. **当前状态**：代码未实现 SM 预测，只做简单的优先级调度
2. **数据准备**：kernel_info.csv 中的 SM 数据已准备好，但未使用
3. **推荐方案**：先实现方法 1（查表），简单且准确
4. **实施难度**：低（约 100 行代码）
5. **关键函数**：需要实现 `get_operator_sm_requirement()` 和 `get_current_sm_usage()`
