# 算子粒度 SM 分配方案对比总结

## 问题描述

H100 GPU 有 114 个 SM，当两个大 GEMM 算子（各需 64 SM）同时运行时，总需求 128 SM > 114 SM，导致严重的资源争抢，性能下降超过 100%。

现有方案只有两个 Green Context（HP: 64 SM, BE: 48 SM），粒度太粗，无法解决算子级别的资源抢占问题。

---

## 方案 1：多流预分配 + 静态 SM 绑定

### 核心思想
创建多个 Green Context（如 4 个），每个绑定固定数量的 SM，算子按 Round-robin 策略分配到不同的流上执行。

```
GPU (114 SMs)
├── GC0 (HP): 40 SMs → Stream0
├── GC1 (BE): 24 SMs → Stream1
├── GC2 (BE): 24 SMs → Stream2
└── GC3 (BE): 24 SMs → Stream3
```

### 优势
- ✅ **实现简单**：基于现有框架，2 周可完成
- ✅ **性能可预测**：静态分配，无运行时开销
- ✅ **硬件隔离**：完全避免 SM 争抢
- ✅ **风险低**：改动小，易测试

### 痛点
- ⚠️ **资源碎片化**：小算子（8 SM）在大 GC（24 SM）上浪费资源
- ⚠️ **负载不均衡**：Round-robin 无法感知算子实际需求
- ⚠️ **配置调优困难**：需要手动调整 GC 数量和 SM 分配
- ⚠️ **Context 切换开销**：每次算子执行前需切换（~5 us）
- ⚠️ **cuBLAS Handle 管理**：每个 GC 需要独立 handle（~10 MB 显存）

### 实施要点
```cpp
// 1. 环境变量配置
ORION_NUM_GREEN_CONTEXTS=4
ORION_GC_SM_COUNTS="40,24,24,24"

// 2. 数据结构
std::vector<CUgreenCtx> green_ctxs_;
std::vector<cudaStream_t> gc_streams_;
std::vector<cublasHandle_t> cublas_handles_;

// 3. 调度逻辑
int select_stream(int client_idx) {
    if (client_idx == 0) return 0;  // HP → GC0
    return 1 + (next_idx++ % 3);    // BE → GC1/2/3 轮询
}
```

---

## 方案 2：动态 SM 分配 + 资源争用检测

### 核心思想
拦截算子后，查询其 SM 需求，检查当前 GPU SM 占用情况，动态创建或复用 Green Context，避免资源争用。

```
调度流程：
1. 拦截算子 Op
2. 查询 Op 的 SM 需求（从 kernel_info.csv）
3. 检查当前 SM 占用
4. if (当前占用 + Op需求 > 114):
       等待资源释放
   else:
       分配 SM 并执行
```

### 优势
- ✅ **资源利用率高**：按需分配，避免浪费
- ✅ **自适应**：无需手动配置，自动适应工作负载
- ✅ **灵活性强**：支持任意 SM 需求的算子
- ✅ **避免争用**：主动检测并避免 SM 过度订阅

### 痛点
- ❌ **实现复杂**：需要 SM 跟踪器、GC 管理器、调度器三个模块（~1500 行代码）
- ❌ **SM 占用跟踪不准确**：CUDA 无实时查询 API，只能估算
- ❌ **Kernel 完成时间不可预测**：依赖异步回调，有延迟
- ❌ **GC 创建/销毁开销大**：每次创建 ~2 ms，需要池化机制
- ❌ **并发控制复杂**：多线程竞争，死锁风险
- ❌ **依赖 Profile 数据**：需要预先 profiling 每个 kernel 的 SM 需求
- ❌ **调度延迟增加**：从 ~10 us 增加到 ~50 us

### 实施要点
```cpp
// 1. SM 资源跟踪器
class SMResourceTracker {
    int get_current_sm_usage();
    bool will_contend(int new_requirement);
};

// 2. 动态 GC 管理器
class DynamicGreenContextManager {
    GCAllocation* allocate_for_operator(int sm_requirement);
    void release(GCAllocation* alloc);
};

// 3. 调度逻辑
if (sm_tracker.will_contend(op_sm_requirement)) {
    continue;  // 等待
}
auto* gc = gc_manager.allocate_for_operator(op_sm_requirement);
execute_operation(op, gc->stream, gc->cublas_handle);
```

---

## 对比分析

| 维度 | 方案 1 | 方案 2 |
|------|--------|--------|
| **实现复杂度** | ✅ 简单（~500 行） | ❌ 复杂（~1500 行） |
| **开发周期** | ✅ 1-2 周 | ❌ 4-6 周 |
| **SM 利用率** | ⚠️ 中等（60-85%） | ✅ 高（85-90%） |
| **调度延迟** | ✅ 低（~10 us） | ⚠️ 中等（~50 us） |
| **配置复杂度** | ⚠️ 需手动调优 | ✅ 自动适应 |
| **可预测性** | ✅ 高 | ⚠️ 中等 |
| **稳定性风险** | ✅ 低 | ⚠️ 中（并发 bug） |
| **维护成本** | ✅ 低 | ❌ 高 |
| **适用场景** | 固定工作负载 | 动态工作负载 |

### 性能对比（理论）

**场景 1：固定工作负载（1 HP + 3 BE，大 GEMM）**
- 方案 1：SM 利用率 85%，HP 延迟基准，吞吐量基准
- 方案 2：SM 利用率 90%，HP 延迟 +40 us，吞吐量 +5%
- **推荐**：方案 1（简单且性能足够）

**场景 2：混合工作负载（大/中/小 GEMM 混合）**
- 方案 1：SM 利用率 60%（碎片化），吞吐量基准
- 方案 2：SM 利用率 85%，吞吐量 +40%
- **推荐**：方案 2（资源利用率显著提升）

**场景 3：高并发（8+ 客户端）**
- 方案 1：SM 利用率 70%（GC 数量受限）
- 方案 2：SM 利用率 90%，吞吐量 +30%
- **推荐**：方案 2（更好的并发支持）

---

## 推荐实施路线

### 阶段 1（短期，2 周）：方案 1 - 快速验证

**理由**：
- 快速验证算子粒度 SM 分配的效果
- 风险可控，改动小
- 适合当前固定工作负载场景

**配置建议**：
```bash
# 1 HP + 3 BE 场景
ORION_NUM_GREEN_CONTEXTS=4
ORION_GC_SM_COUNTS="40,24,24,24"
```

**成功标准**：
- SM 隔离验证通过
- HP 延迟 ≤ 当前方案 × 1.1
- 总吞吐量 ≥ 当前方案 × 1.3

### 阶段 2（中期，1 个月）：方案 1 优化

在方案 1 基础上增加：
- 基于 SM 需求的智能 Stream 选择
- 负载均衡（监控队列长度）
- 自适应配置推荐

### 阶段 3（长期，3 个月）：方案 2 评估

如果出现以下情况，考虑方案 2：
- 工作负载高度动态
- 需要支持 10+ 客户端
- SM 利用率 <70%

---

## 实施清单（方案 1）

### Week 1：核心功能开发
- [ ] 扩展 `GreenCtxConfig` 支持多 GC
- [ ] 实现 `init_green_contexts_pool()`
- [ ] 实现 `select_stream_for_operation()`
- [ ] 修改 `run()` 主循环
- [ ] 单元测试

### Week 2：测试与优化
- [ ] SM 隔离验证（`test_sm_isolation.cu`）
- [ ] 功能测试（`test_orion_blocking.py`）
- [ ] 性能测试（对比不同配置）
- [ ] 错误处理和 fallback
- [ ] 文档更新

---

## 总结

**方案 1（推荐）**：多流预分配 + 静态 SM 绑定
- 适合固定工作负载，快速验证，风险低
- 2 周可完成，性能提升明显

**方案 2（长期）**：动态 SM 分配 + 资源争用检测
- 适合动态工作负载，资源利用率高
- 实现复杂，需 4-6 周，稳定性风险较高

**建议**：先实施方案 1，验证效果后再评估是否需要方案 2。

