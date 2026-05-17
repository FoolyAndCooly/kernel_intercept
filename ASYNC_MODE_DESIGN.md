# 异步模式改造方案

## 一、改造目标

将算子间隔从 **2-10μs** 降低到 **~150ns**（可忽略不计）

## 二、核心策略：三级异步模式

### Level 0: 同步模式（当前默认）
- 客户端线程：入队 → 忙等 worker 执行 → 返回
- 算子间隔：2-10μs
- 适用场景：调试、需要精确错误处理

### Level 1: Worker 异步（推荐）
- 客户端线程：入队 → **立即返回**（假设成功）
- Worker 线程：异步执行真实 CUDA API
- 算子间隔：~150ns
- 适用场景：生产环境、高吞吐量

### Level 2: 直通异步（极致性能）
- 客户端线程：**直接调用真实 CUDA API** → 立即返回
- 完全绕过队列和 worker
- 算子间隔：~50ns
- 适用场景：单 client、无需调度

## 三、实现细节

### 3.1 环境变量控制

```bash
export ORION_ASYNC_MODE=0  # 同步模式（默认）
export ORION_ASYNC_MODE=1  # Worker 异步
export ORION_ASYNC_MODE=2  # 直通异步
```

### 3.2 代码改动

#### 文件：`src/cuda_intercept.cpp`

**改动 1：cudaLaunchKernel wrapper**

```cpp
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    SAFE_PASSTHROUGH(cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);

    if (!is_capture_enabled()) 
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    int async_mode = get_async_mode_internal();
    
    // Level 2: 直通异步（完全绕过调度器）
    if (async_mode == 2) {
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }

    int client_idx = resolve_client_idx(stream);
    if (client_idx < 0) 
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH);
    if (!op) return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);

    KernelLaunchParams kp;
    kp.func = func;
    kp.gridDim = gridDim;
    kp.blockDim = blockDim;
    kp.sharedMem = sharedMem;
    kp.stream = stream;
    kp.original_args = args;
    kp.use_deep_copy = false;  // CUDA 内部会拷贝，无需我们拷贝

    op->params = std::move(kp);
    enqueue_operation(op);

    // Level 1: Worker 异步（立即返回，不等待）
    if (async_mode == 1) {
        return cudaSuccess;  // 假设成功，错误延迟到 cudaDeviceSynchronize
    }

    // Level 0: 同步模式（忙等）
    wait_operation(op);
    return op->result;
}
```

**改动 2：错误处理机制**

异步模式下，kernel launch 错误会延迟到下一个同步点才能检测到。需要：

```cpp
// 在 worker 线程中记录错误
void Scheduler::run_worker(int client_idx) {
    while (running_) {
        auto op = q->try_pop();
        if (!op) { yield(); continue; }

        cudaError_t err = execute_operation(op, exec_stream, cublas_handle);
        
        // 异步模式：记录错误但不阻塞客户端
        if (err != cudaSuccess) {
            LOG_ERROR("Worker %d: op %lu failed with error %d", 
                      client_idx, op->op_id, (int)err);
            // 可选：设置全局错误标志
            g_last_error[client_idx].store(err);
        }

        op->mark_completed(err);
    }
}

// 在 cudaDeviceSynchronize 中检查累积的错误
cudaError_t cudaDeviceSynchronize() {
    SAFE_PASSTHROUGH(cudaDeviceSynchronize);
    
    int client_idx = resolve_client_idx();
    if (client_idx >= 0) {
        // 同步 worker 队列（确保所有 op 都被执行）
        orion_sync_client_stream(client_idx);
    }
    
    // 调用真实的 cudaDeviceSynchronize
    cudaError_t err = g_real_funcs.cudaDeviceSynchronize();
    
    // 检查是否有累积的错误
    if (client_idx >= 0) {
        cudaError_t async_err = g_last_error[client_idx].load();
        if (async_err != cudaSuccess) {
            g_last_error[client_idx].store(cudaSuccess);  // 清除错误
            return async_err;  // 返回第一个错误
        }
    }
    
    return err;
}
```

### 3.3 性能对比（理论估算）

| 模式 | 算子间隔 | 吞吐量 | 错误检测 | 适用场景 |
|------|---------|--------|---------|---------|
| Level 0 (同步) | 2-10μs | 100K-500K ops/s | 立即 | 调试 |
| Level 1 (Worker异步) | ~150ns | 6M ops/s | 延迟到sync | 生产 |
| Level 2 (直通异步) | ~50ns | 20M ops/s | 延迟到sync | 单client |

### 3.4 兼容性保证

- **默认行为不变**：`ORION_ASYNC_MODE` 未设置时保持同步模式
- **向后兼容**：现有测试脚本无需修改
- **渐进式启用**：可以先在单 client 场景测试 Level 1，稳定后推广

## 四、实施步骤

1. **gpu_capture.cpp**：添加 `g_last_error[]` 数组
2. **cuda_intercept.cpp**：修改 `cudaLaunchKernel` 支持三级模式
3. **cuda_intercept.cpp**：修改 `cudaDeviceSynchronize` 检查累积错误
4. **scheduler.cpp**：Worker 线程记录错误到 `g_last_error[]`
5. **测试验证**：
   - 单元测试：验证错误传播正确
   - 性能测试：测量算子间隔
   - 正确性测试：对比同步/异步结果一致性

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 错误延迟检测 | 调试困难 | 提供详细日志，记录错误发生的 op_id |
| 参数生命周期 | 潜在崩溃 | 依赖 CUDA 内部拷贝（已验证安全） |
| 多线程竞态 | 数据损坏 | 使用 atomic 操作，无锁队列 |

## 六、预期收益

- **单 client 吞吐量**：提升 10-60x
- **多 client 并发**：保持不变（已经是并行的）
- **端到端延迟**：降低 2-10μs（对小 kernel 显著）
