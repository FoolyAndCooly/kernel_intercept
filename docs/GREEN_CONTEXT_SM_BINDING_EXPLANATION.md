# Green Context SM 绑定机制详解

## 核心问题

**Q: Green Context 使用 30 个 SM 后，是把这 30 个 SM 直接划到这个流上了（静态绑定），还是只是说提交到这个流会分配 30 个，但不确定是哪 30 个（动态分配）？**

---

## 答案：静态绑定到特定的 SM

**Green Context 是将特定的 30 个 SM 静态绑定到该 context，只有提交到这个 context 的 stream 才能使用这些特定的 SM。**

---

## 详细解释

### 1. 静态 SM 分区（Static SM Partitioning）

根据 CUDA 文档：

> "Using green contexts, one could partition the GPU's SMs, so that green context A, targeted by kernel A, has access to **some SMs** of the GPU, while green context B, targeted by kernel B, has access to the **remaining SMs**."

> "A green context provisioned with N SMs during its creation can only use **these specific N SMs**."

**关键点**：
- ✅ Green Context 创建时，会分配**特定的 N 个 SM**
- ✅ 这些 SM 是**固定的、具体的物理 SM**（如 SM 0-29）
- ✅ 只有该 Green Context 的 stream 才能使用这些 SM
- ✅ 其他 Green Context 无法使用这些 SM（除非 oversubscription）

### 2. 与 MPS Active Thread Percentage 的对比

CUDA 文档明确对比了 Green Context 和 MPS：

> "With MPS, the active thread percentage signifies that a given client application cannot use more than x% of a GPU's SMs, let that be N SMs. However, **these SMs can be any N SMs of the GPU, which can also vary over time**."

> "On the other hand, a green context provisioned with N SMs during its creation can only use **these specific N SMs**."

**对比**：

| 机制 | SM 分配方式 | SM 是否固定 |
|------|------------|------------|
| **Green Context** | 特定的 N 个 SM（如 SM 0-29） | ✅ 固定 |
| **MPS Active Thread %** | 任意 N 个 SM，可能变化 | ❌ 动态 |

### 3. 实际例子

假设 H100 有 114 个 SM（编号 SM 0 到 SM 113）：

```cpp
// 创建两个 Green Context
cudaDevResource result[2] = {{}, {}};
cudaDevSmResourceGroupParams group_params[2] = {
    {.smCount=30, ...},  // GC0: 30 SMs
    {.smCount=84, ...}   // GC1: 84 SMs
};

cudaDevSmResourceSplit(&result[0], 2, &initial_GPU_SM_resources,
                       nullptr, 0, &group_params[0]);

cudaExecutionContext_t gc0, gc1;
cudaGreenCtxCreate(&gc0, desc0, 0, 0);  // GC0
cudaGreenCtxCreate(&gc1, desc1, 0, 0);  // GC1
```

**实际分配（示例）**：
- **GC0**：绑定到 SM 0-29（30 个 SM，固定）
- **GC1**：绑定到 SM 30-113（84 个 SM，固定）

**行为**：
```cpp
cudaStream_t stream0, stream1;
cudaExecutionCtxStreamCreate(&stream0, gc0, cudaStreamDefault, 0);
cudaExecutionCtxStreamCreate(&stream1, gc1, cudaStreamDefault, 0);

// Kernel A 提交到 stream0
kernel_A<<<grid, block, 0, stream0>>>();
// → 只能在 SM 0-29 上运行，永远不会使用 SM 30-113

// Kernel B 提交到 stream1
kernel_B<<<grid, block, 0, stream1>>>();
// → 只能在 SM 30-113 上运行，永远不会使用 SM 0-29
```

### 4. 如何验证 SM 绑定

**方法 1：使用 Nsight Compute**

Nsight Compute 的 "Green Context Resources" 视图会显示每个 Green Context 绑定的具体 SM：

```
Green Context 0:
  SM Bitmask: [████████████████████████████████                                                                                                ]
  Provisioned SMs: 0-29 (30 SMs)

Green Context 1:
  SM Bitmask: [                                ████████████████████████████████████████████████████████████████████████████████████████████████]
  Provisioned SMs: 30-113 (84 SMs)
```

**方法 2：使用 CUDA API 查询**

```cpp
// 查询 Green Context 的 SM 资源
cudaDevResource gc_sm_resource = {};
cudaExecutionCtxGetDevResource(gc0, &gc_sm_resource, cudaDevResourceTypeSm);

std::cout << "GC0 has " << gc_sm_resource.sm.smCount << " SMs" << std::endl;
// 输出：GC0 has 30 SMs

// 注意：API 不直接返回具体的 SM 编号，但可以通过 Nsight 工具查看
```

### 5. SM Oversubscription（重叠）

Green Context 允许 SM 重叠（oversubscription），但这是**显式的、可控的**：

```cpp
// 创建 3 个 Green Context，部分重叠
// GC0: SM 0-63   (64 SMs)
// GC1: SM 64-113 (50 SMs)
// GC2: SM 0-31   (32 SMs, 与 GC0 重叠)

// 行为：
// - GC0 和 GC1 可以并发（SM 不重叠）
// - GC0 和 GC2 不能并发（SM 0-31 重叠，会冲突）
// - GC1 和 GC2 可以并发（SM 不重叠）
```

**关键点**：
- ✅ 重叠是在创建时确定的，不是运行时动态的
- ✅ 重叠的 SM 仍然是特定的物理 SM
- ⚠️ 重叠的 GC 并发时会冲突

---

## 总结

### 问题答案

**Green Context 使用 30 个 SM 后：**

✅ **是把这 30 个 SM 直接划到这个 context 上了（静态绑定）**
- 这 30 个 SM 是特定的物理 SM（如 SM 0-29）
- 只有该 Green Context 的 stream 才能使用这些 SM
- 其他 Green Context 无法使用这些 SM（除非显式 oversubscription）

❌ **不是动态分配 30 个不确定的 SM**
- 不是"提交时分配任意 30 个 SM"
- 不是"每次运行可能使用不同的 SM"
- 不是"运行时竞争 SM 资源"

### 设计含义

这意味着你的**方案 2 的原始设计是错误的**：

**错误设计**：
```
创建 3 个 GC，每个 64 SMs，总和 192 > 114
假设调度器控制并发就不会冲突
```

**为什么错误**：
- 3 个 GC 各 64 SMs，总和 192 > 114，必然有 SM 重叠
- 即使调度器控制"同时只运行 2 个 GC"，如果这 2 个 GC 的 SM 有重叠，仍会冲突
- Green Context 不是"虚拟 SM 池"，而是"静态 SM 分区"

**正确设计**：
```
创建 2 个不重叠的 GC（56+56=112 SMs）
创建 5 个 stream，共享这 2 个 GC
调度器跟踪 GC 级别的并发（确保不同 GC 的 stream 才能并发）
```

### 类比

**Green Context 的 SM 分配类似于**：
- ✅ 硬盘分区（每个分区有固定的扇区）
- ✅ 内存分段（每个段有固定的地址范围）
- ✅ CPU 核心绑定（进程绑定到特定的 CPU 核心）

**不类似于**：
- ❌ 虚拟内存（物理页可以动态映射）
- ❌ 线程池（线程可以执行任意任务）
- ❌ 资源配额（限制使用量但不限制具体资源）

---

## 参考文档

CUDA Programming Guide - Section 4.6: Green Contexts

关键引用：
> "A green context provisioned with N SMs during its creation can only use these specific N SMs."

> "On the other hand, a green context provisioned with N SMs during its creation can only use these specific N SMs."
