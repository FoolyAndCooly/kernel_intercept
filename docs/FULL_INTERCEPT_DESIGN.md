# 完全拦截服务算子 —— 修改方案

## 1. 背景与目标

### 1.1 当前状态

以 `orion_gpt_vgg_1be_trace.json`（HP=GPT, BE=VGG, 1HP+1BE, ISOLATED 模式）为基线，trace 显示：

- GPU 上有 5 条活跃流，分属 3 个 CUDA context：
  - `ctx=1` (torch primary): stream 39（HP torch 原生流，118 个 kernel）+ stream 43（BE torch 原生流，158 个 kernel）
  - `ctx=2` (HP Green Context): stream 20（226 个 cuBLAS/cuBLASLt kernel）
  - `ctx=3` (BE Green Context): stream 21（12 个 GEMM kernel）

- 时间线（相对 t0）：
  - HP 0 .. 12980 µs
  - BE 7059 .. 23858 µs（晚 7 ms 才启动）
  - HP∩BE 真正 GPU 并发 ≈ 115 µs，不足整段的 1%

### 1.2 错开的根因

1. **API 覆盖不全**：只拦 `cudaLaunchKernel`，`cudaLaunchKernelExC` / `cuLaunchKernel` / `cuLaunchKernelEx` 漏网 → 相关 kernel 直接落到 torch 原生流 ctx 1，GC 无法隔离。
2. **线程归属模型错位**：`tl_client_idx` 是 thread-local，cuDNN 内部 worker 线程、PyTorch CachingAllocator 后台线程没有 client_idx → 它们发的 kernel 走 `client_idx < 0` fallback，绕过调度器。
3. **调度器单线程 + 同步提交**：`Scheduler::run` 只有一根线程轮询所有队列，client 侧每个 op 都 `wait_operation` 忙等；HP 和 BE 的 op 在 host 侧被串行分发。
4. **Capture 启用时机晚 / 冷启动计入主体**：`test_orion_blocking.py` warmup 在 capture 开启之前；cuDNN 第一次用 workspace/handle 时冷启动 7 ms 整个被算进 BE 的主体时间。

### 1.3 目标

- BE 总时长从 ~23 ms 压到 ~13 ms（和 HP 对齐）。
- HP∩BE GPU 并发时间从 115 µs 拉到至少 8 ms（重叠率 > 60%）。
- 所有 kernel launch 都经过调度器，stream 39/43 上不再有运行时 workload kernel（只保留一次性初始化）。

---

## 2. 方案总览

分四步落地，按"收益/风险比"从高到低排序：

| 步骤 | 改动量 | 改动范围 | 预期效果 | 风险 |
|---|---|---|---|---|
| **Step A** 冷启动前移 + capture 时机修正 | 小 | `python/test_orion_blocking.py` + 1 个 C API | BE 启动从 7 ms 提前到 < 500 µs | 低 |
| **Step B** stream→client 映射 + `cudnnSetStream` / `cublasSetStream` 强关联 | 中 | `src/gpu_capture.cpp/.h`, `src/cudnn_intercept.cpp`, `src/cublas_intercept.cpp`, Python | BE 的 cuDNN conv 主体段（15 ms）进入调度器，落到 BE GC stream | 中（需要覆盖 cuDNN 子线程） |
| **Step C** 补全 launch API（`cudaLaunchKernelExC` / `cuLaunchKernel` / `cuLaunchKernelEx`） | 中 | `src/cuda_intercept.cpp`, `include/gpu_capture.h`, `include/common.h` | 清掉 torch 原生流上的剩余漏网 kernel | 中（需正确处理 attrs 深拷贝） |
| **Step D** 多 dispatcher + 异步提交 + event 等完 | 大 | `src/scheduler.cpp`, `include/gpu_capture.h`, `include/scheduler.h` | host 端 per-op overhead 从 5–20 µs 降到 <1 µs，HP/BE 真正并行分发 | 高（改调度核心） |

本文档按顺序给出每一步的**具体改动点、代码骨架、验证方式和回滚路径**。Step A+B 是最低承诺版本，一晚上能做完并见效；Step C+D 是持续优化。

---

## 3. Step A：冷启动前移 + capture 时机修正

### 3.1 问题定位

`python/test_orion_blocking.py:253-262` 的当前时序：

```python
lib.orion_start_scheduler_thread()          # ①
# ... 构造 models、inputs
for i in range(num_clients):                 # ②【warmup，此时 capture 未启用】
    with torch.no_grad():
        _ = models[i](inputs[i])
torch.cuda.synchronize()
# 进入 profiler
with torch.profiler.profile(...) as prof:
    start.set()                              # ③ worker 线程才真正开始
```

warmup ② 发生时 `g_capture_state.enabled == false` → 所有 cuDNN handle、cuBLAS workspace 是绑在**torch primary context 1** 上创建的，不是 GC context。进入 ③ 时，第一次 BE conv 触发 cuDNN 在 GC 上重新绑 handle、分配 workspace → 7 ms 冷启动全部计入 BE 主体。

### 3.2 改动

#### 3.2.1 C API：显式开关 capture

`include/gpu_capture.h` 已有 `set_capture_enabled`（通过 `orion_set_capture` 暴露）。确认 Python 侧能调到即可；若缺失 C 导出，在 `src/gpu_capture.cpp` 尾部补：

```cpp
extern "C" {
    int orion_set_capture(int enabled) {
        orion::set_capture_enabled(enabled != 0);
        return 0;
    }
}
```

`test_orion_blocking.py` 的 `setup_scheduler_lib()` 里绑定：

```python
lib.orion_set_capture.argtypes = [ctypes.c_int]
lib.orion_set_capture.restype = ctypes.c_int
```

#### 3.2.2 Python 时序调整

改 `test_orion_blocking.py:253` 前后的流程：

```python
# 1. 启动调度器线程（此时 capture 仍未启用）
lib.orion_start_scheduler_thread()

# 2. 【新增】开启 capture —— 必须在 warmup 之前
lib.orion_set_capture(1)

# 3. 每个 client 的 worker 线程内：先设 client_idx，再做 warmup
def worker(idx):
    lib.orion_set_client_idx(idx)
    with torch.cuda.stream(streams[idx]):
        with torch.no_grad():
            _ = models[idx](inputs[idx])        # ← warmup，此时 capture 已开，cuDNN/cuBLAS handle 会绑到 GC
    torch.cuda.synchronize()
    barrier.wait()                               # 预热完才进 barrier
    # 真正测时
    t0 = time.time()
    ...
```

这样每个 worker 先在自己的 client_idx 下跑一次完整前向，把 cuDNN handle、cuBLAS workspace、CUDA module 缓存等全部在 capture 开启态建好；再同步 barrier 进入计时段。

#### 3.2.3 调度器侧 dummy prime

在 `src/scheduler.cpp::init_green_contexts` 成功后加一次 dummy kernel，确保每个 GC context 的 module loader / JIT cache 预热：

```cpp
// scheduler.cpp, 在 init_green_contexts() 成功返回前
for (int ctx_idx = 0; ctx_idx < (int)cuda_ctxs_.size(); ++ctx_idx) {
    cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
    void* dummy = nullptr;
    cudaMalloc(&dummy, 16);
    cudaMemsetAsync(dummy, 0, 16, ctx_idx == 0 ? hp_gc_stream_ : be_gc_streams_[0]);
    cudaFree(dummy);
}
cuCtxSetCurrent(cuda_ctxs_[0]);
current_ctx_idx_ = 0;
```

### 3.3 验证

- 重跑 `run.sh`，新 trace 对比 BE 的首个 kernel 时间 相对 HP 的偏移应 < 500 µs。
- `logs/scheduler.log` 里 "cuBLASLt workspace allocation" / "cuDNN plan cache miss" 这类提示不应出现在 profiler 窗口内。

### 3.4 回滚

只改了 Python 和一处调度器初始化，回滚只需 git revert 两个提交。

---

## 4. Step B：stream→client 映射 + 库 handle 绑定追踪

### 4.1 问题定位

- `src/gpu_capture.cpp:107 get_current_client_idx()` 只看 `tl_client_idx`；cuDNN `cudnnExecutionPlan` 内部会起 worker 线程实际发 `cudaLaunchKernel`，那个线程没 `tl_client_idx` → `cuda_intercept.cpp:841` 直接 fallback，不入队。
- trace 证据：stream 43 上 126 次 `cudaLaunchKernel` 来自 tid 1166008320，不是调度器线程也不是注册过的 client 主线程。

### 4.2 改动

#### 4.2.1 引入 stream→client、handle→client 两级映射

`include/gpu_capture.h` 追加接口：

```cpp
void orion_register_client_stream(int client_idx, cudaStream_t stream);
void orion_register_client_cudnn_handle(int client_idx, cudnnHandle_t h);
void orion_register_client_cublas_handle(int client_idx, cublasHandle_t h);
void orion_register_client_cublaslt_handle(int client_idx, cublasLtHandle_t h);

// 统一查询入口：先看 tl_client_idx，再看 stream，再看 handle
int resolve_client_idx(cudaStream_t stream = nullptr, void* handle = nullptr);
```

`src/gpu_capture.cpp` 实现骨架：

```cpp
static std::unordered_map<cudaStream_t, int> g_stream_to_client;
static std::unordered_map<void*, int>        g_handle_to_client;
static std::shared_mutex                     g_map_mu;

void orion_register_client_stream(int client_idx, cudaStream_t stream) {
    std::unique_lock lk(g_map_mu);
    g_stream_to_client[stream] = client_idx;
}
// 其余 register 同理

int resolve_client_idx(cudaStream_t stream, void* handle) {
    if (tl_client_idx >= 0) return tl_client_idx;   // fast path
    std::shared_lock lk(g_map_mu);
    if (stream) {
        auto it = g_stream_to_client.find(stream);
        if (it != g_stream_to_client.end()) return it->second;
    }
    if (handle) {
        auto it = g_handle_to_client.find(handle);
        if (it != g_handle_to_client.end()) return it->second;
    }
    return -1;
}
```

#### 4.2.2 修改所有拦截 wrapper 的 client_idx 查询

`src/cuda_intercept.cpp` 现有：

```cpp
int client_idx = get_current_client_idx();            // 只看 tl
```

改为：

```cpp
int client_idx = resolve_client_idx(stream /* 可选 */);
```

`src/cudnn_intercept.cpp` / `src/cublas_intercept.cpp` 的每个 wrapper 用 handle 参数：

```cpp
int client_idx = resolve_client_idx(/*stream=*/nullptr, /*handle=*/(void*)handle);
```

#### 4.2.3 拦截 `cudnnSetStream` / `cublasSetStream` 维护 handle→stream 关系

`src/cudnn_intercept.cpp` 新增：

```cpp
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
    SAFE_PASSTHROUGH_CUDNN(cudnnSetStream, handle, streamId);
    // 若 streamId 已绑定到某 client，handle 也跟随
    int client = resolve_client_idx(streamId);
    if (client >= 0) orion_register_client_cudnn_handle(client, handle);
    return real_cudnnSetStream(handle, streamId);
}
```

`cublasSetStream_v2` 已有拦截（`src/cublas_intercept.cpp:1283`），在那里加同样的登记。

#### 4.2.4 Python 侧注册

`test_orion_blocking.py`：

```python
streams = [torch.cuda.Stream() for _ in range(num_clients)]
for i, s in enumerate(streams):
    lib.orion_register_client_stream(i, ctypes.c_void_p(s.cuda_stream))
```

`cudaStream_t` 在 torch 里是 `s.cuda_stream`（整数）。

### 4.3 验证

- 新 trace 里 stream 43 上的 kernel 数应 ≈ 0（只剩启动期间个位数）。
- `logs/scheduler.log` 搜索 "`client_idx < 0, fallback`"，数量应从几百条降到 0。
- stream 20 / 21（GC stream）上的 kernel 数增加到接近原先 stream 39+43 的总和。

### 4.4 风险 & 回滚

- cuDNN 版本差异：`cudnnSetStream` 签名稳定，风险低；但某些 cuDNN v9 plan 会跳过 `cudnnSetStream` 直接用 plan 内部绑定的 stream，需追加拦 `cudnnExecutionPlanExecute`。
- handle→client 映射 race：多 client 共用同一个 cuBLAS handle（PyTorch 默认每 device 一个）会互相覆盖。解决：为每个 client 强制独立 handle（调度器已经有 `cublas_handles_[ctx_idx]`），Python 里不要让 torch 复用。
- 回滚：`resolve_client_idx` 提供 `tl_client_idx` fast path，退化到旧语义只需在查询入口加 `return tl_client_idx`。

---

## 5. Step C：补全 Launch API 覆盖

### 5.1 问题定位

trace 里来源线程统计：

```
stream 20: 126× cudaLaunchKernelExC（调度器线程，passthrough 正常）
stream 14 (另一 profile): 117× cudaLaunchKernelExC + 81× cudaLaunchKernel + 3× cuLaunchKernel
```

漏网 API：

- `cudaLaunchKernelExC` — CUDA 11.6+ 的新 runtime API，cuBLASLt / cuDNN v9 / PyTorch 2.x 都在用
- `cuLaunchKernel` / `cuLaunchKernelEx` — driver API，部分 CUDA graph 和 cuBLASLt JIT kernel 走这条

### 5.2 改动

#### 5.2.1 新 OperationType 和参数结构

`include/common.h::OperationType` 追加：

```cpp
KERNEL_LAUNCH_EX,       // cudaLaunchKernelExC
KERNEL_LAUNCH_DRV,      // cuLaunchKernel / cuLaunchKernelEx
```

`op_type_name` 补对应 case。

`include/gpu_capture.h` 增加 params struct：

```cpp
struct KernelLaunchExParams {
    // 从 cudaLaunchConfig_t 深拷贝出来（attrs 指针是临时的）
    cudaLaunchConfig_t config;            // gridDim / blockDim / dynamicSmemBytes / stream
    std::vector<cudaLaunchAttribute> attrs;   // 深拷贝
    const void* func;
    void** original_args;
    // ...
};

struct DrvKernelLaunchParams {
    CUfunction func;
    unsigned int gx, gy, gz, bx, by, bz;
    unsigned int shmem;
    CUstream stream;
    void** kernel_params;     // 指针数组，同步模式下直接用
    void** extra;
};
```

并加入 `OperationRecord::params` 的 `std::variant`。

#### 5.2.2 Wrapper 实现

`src/cuda_intercept.cpp` 追加：

```cpp
using cudaLaunchKernelExC_t = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
using cuLaunchKernel_t      = CUresult (*)(CUfunction, unsigned, unsigned, unsigned,
                                            unsigned, unsigned, unsigned, unsigned,
                                            CUstream, void**, void**);

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config,
                                const void* func, void** args) {
    using namespace orion;
    SAFE_PASSTHROUGH(cudaLaunchKernelExC, config, func, args);
    if (!is_capture_enabled()) return real_cudaLaunchKernelExC(config, func, args);

    int client_idx = resolve_client_idx(config->stream);
    if (client_idx < 0) return real_cudaLaunchKernelExC(config, func, args);

    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH_EX);
    if (!op) return real_cudaLaunchKernelExC(config, func, args);

    KernelLaunchExParams kp;
    kp.config = *config;
    kp.attrs.assign(config->attrs, config->attrs + config->numAttrs);
    kp.config.attrs = kp.attrs.data();
    kp.func = func;
    kp.original_args = args;
    op->params = std::move(kp);
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                        unsigned bx, unsigned by, unsigned bz, unsigned shmem,
                        CUstream hs, void** kp, void** extra) {
    // 同样模式：SAFE_PASSTHROUGH_DRV → resolve_client_idx(hs) → 入队 KERNEL_LAUNCH_DRV
}
```

执行端 `execute_kernel_launch_ex` / `_drv` 镜像现有 `execute_kernel_launch`，用 `scheduler_stream` 覆盖 `config.stream` / `hs`。

#### 5.2.3 初始化真实函数指针

`init_real_functions()` 里补：

```cpp
g_real_funcs.cudaLaunchKernelExC = (cudaLaunchKernelExC_t)get_cuda_func("cudaLaunchKernelExC");
g_real_funcs.cuLaunchKernel      = (cuLaunchKernel_t)dlsym(RTLD_NEXT, "cuLaunchKernel");
```

### 5.3 验证

- 新 trace 里 stream 39/43 上 `cudaLaunchKernel*` 事件数应 ≈ 0。
- 所有 kernel 的 correlation id 都能在 `cuda_runtime` 事件里追到调度器线程 tid。

### 5.4 风险

- `cudaLaunchConfig_t` 的 `attrs` 在 CUDA 12.x 里字段可能扩展；保持 `cudaLaunchAttribute` 值拷贝即可。
- `cuLaunchKernel` 的 `extra` 参数是 key-value 可变数组，默认 `NULL`；如非 null 需深拷贝（实际 PyTorch/cuBLAS 都用 `kernel_params`，暂不处理 `extra`，遇到时 fallback）。

---

## 6. Step D：多 dispatcher + 异步提交 + CUDA event 等完

### 6.1 问题定位

现在的热点路径（`src/scheduler.cpp:1012-1065`）：

```
client 线程          scheduler 线程
  enqueue ─────────▶
                      pop
                      execute (5-20 µs)
                      mark_completed ◀
  wait (busy spin)   ─┘
  resume
```

每个 op 5-20 µs host overhead，HP 和 BE 的 op 在**单队列轮询**里串行执行。对 ~400 ops/iter 的 workload，host 侧就能多花 2-4 ms。

### 6.2 改动

#### 6.2.1 每 client 一个 dispatcher 线程

`src/scheduler.h`:

```cpp
class Scheduler {
    std::vector<std::thread> dispatcher_threads_;
    // 保留原 thread_ 用于 autotune / drain 特殊路径

    void dispatch_for_client(int client_idx);
};
```

`src/scheduler.cpp::start()` 改为：

```cpp
for (int i = 0; i < num_clients_; ++i) {
    dispatcher_threads_.emplace_back(&Scheduler::dispatch_for_client, this, i);
}
```

`dispatch_for_client(i)`：

```cpp
void Scheduler::dispatch_for_client(int i) {
    tl_is_scheduler_thread = true;

    // ISOLATED 模式下每个线程绑死一个 GC context，避免 run-time cuCtxSetCurrent
    int ctx_idx = (i == 0) ? 0 : 1;
    if (current_mode_ == ExecutionMode::ISOLATED && !cuda_ctxs_.empty()) {
        cuCtxSetCurrent(cuda_ctxs_[ctx_idx]);
    }

    cudaStream_t my_stream =
        (current_mode_ == ExecutionMode::ISOLATED)
            ? (i == 0 ? hp_gc_stream_ : be_gc_streams_[0])
            : (i == 0 ? hp_default_stream_ : be_default_streams_[i - 1]);
    cublasHandle_t my_cublas =
        (current_mode_ == ExecutionMode::ISOLATED)
            ? cublas_handles_[ctx_idx]
            : nullptr;

    while (running_.load()) {
        auto op = g_capture_state.client_queues[i]->try_pop();
        if (!op) { std::this_thread::yield(); continue; }

        // BE 限流：SM 冲突时把 op 放回队头（需要给 ClientQueue 加 push_front）
        if (i > 0 && !orion_should_schedule(op, i)) {
            g_capture_state.client_queues[i]->push_front(op);
            std::this_thread::yield();
            continue;
        }

        cudaError_t err = execute_operation(op, my_stream, my_cublas);
        {
            std::lock_guard<std::mutex> lock(g_orion_state.mutex);
            g_orion_state.seen[i]++;
        }
        op->mark_completed(err);
    }
}
```

效果：HP/BE 各自一个线程，互不争抢队列和 context 切换；ISOLATED 模式下 `cuCtxSetCurrent` 从"每次 op 潜在一次"降到"线程启动时一次"。

#### 6.2.2 异步提交：cuBLAS / cuBLASLt / cuDNN / `cudaLaunchKernelExC`

这些 op 的参数是**已知固定结构体**，可以安全深拷贝。client 入队后立即返回，不 `wait_operation`。

新增 flag 控制：

```cpp
// include/common.h
struct OrionRuntimeConfig {
    bool async_cublas = true;
    bool async_cudnn  = true;
    bool async_launch_ex = true;   // cudaLaunchKernelExC 可异步（config 全拷贝）
    bool async_launch    = false;  // 纯 cudaLaunchKernel 无法深拷参数，保持同步
};
```

`cuda_intercept.cpp::cudaLaunchKernel` 保持同步（`args` 大小未知）。

`cublas_intercept.cpp` 已有 `use_deep_copy` 基础结构，把几个 `wait_operation(op)` 换成：

```cpp
if (g_runtime_config.async_cublas) {
    // 不 wait，立即返回 success（cuBLAS 契约允许失败延后显现）
    return CUBLAS_STATUS_SUCCESS;
}
wait_operation(op);
return op->result_cublas;
```

⚠️ 异步化语义变化：若 op 执行失败，错误会在后续 `cudaStreamSynchronize` 或 `cudaGetLastError` 时显现，与原生 CUDA 行为一致。

#### 6.2.3 `cudaStreamSynchronize` / `cudaDeviceSynchronize` 走 event

`cudaStreamSynchronize(s)` 被拦截时：

```cpp
auto op = create_operation(client_idx, OperationType::STREAM_SYNC);
SyncParams sp; sp.stream = s;
op->params = sp;
enqueue_operation(op);
wait_operation(op);   // ← 这里改成基于 event：
                      //    dispatcher 执行到这个 op 时做
                      //    cudaEventRecord(op->event, my_stream);
                      //    cudaEventSynchronize(op->event);
                      //    然后 mark_completed。
                      // client 侧仍然忙等 completed flag，但实际阻塞在调度器线程对 event 的 wait。
```

更激进的版本：给 `OperationRecord` 加一个 `cudaEvent_t completion_event`，`wait_operation` 改成 `cudaEventSynchronize`，释放 CPU。暂不做，先用现有忙等。

### 6.3 验证

- 在 `scheduler.cpp` 加 `per-client dispatch latency histogram`（op enqueue → mark_completed 时间），改动前后对比 avg/p99。
- 新 trace 里 HP 和 BE 的首个 kernel 时间差 < 100 µs。
- HP 和 BE 的总时长差异 < 1 ms，两者 GPU overlap > 60%。

### 6.4 风险

- 异步化 cuBLAS/cuDNN 后，`torch.cuda.synchronize()` 必须能正确反映所有挂起 op 的完成 → 依赖 Step C 的 `cudaStreamSynchronize` 拦截正确入队并 drain 整个 client queue。
- `orion_should_schedule` 在 dispatcher 线程中调用，可能访问共享状态需加锁（现有实现已经用 `g_orion_state.mutex`，先复用）。
- `ClientQueue` 需要支持 `push_front`，或把限流逻辑改成"暂存一个 pending op"。
- 多 dispatcher 时 `g_orion_state.seen` 的竞争：保留 mutex 即可。

---

## 7. 阶段性里程碑与验收标准

| Milestone | 包含步骤 | 验收标准（基于 1HP+1BE, GPT vs VGG） |
|---|---|---|
| M1 | A | BE 首 kernel 相对 HP 偏移 < 500 µs；BE 总时长 ≤ 16 ms |
| M2 | A + B | stream 39/43 运行时 kernel 数 = 0；BE 总时长 ≤ 14 ms；GPU overlap ≥ 3 ms |
| M3 | A + B + C | trace 里只剩 stream 20 / 21（+ 初始化 stream 167）；所有 kernel 的 correlation 线程都是调度器线程 |
| M4 | A + B + C + D | HP 与 BE 总时长差 ≤ 1 ms；GPU overlap ≥ 8 ms（≥ HP 时长的 60%）；host per-op overhead < 2 µs |

---

## 8. 代码改动清单（备查）

### 必改文件
- `include/common.h` — 新增 OperationType，对应 `op_type_name` case
- `include/gpu_capture.h` — 新增 `KernelLaunchExParams`, `DrvKernelLaunchParams`, 加入 variant；新增 register/resolve API
- `include/scheduler.h` — 新增 `dispatcher_threads_`, `dispatch_for_client`
- `src/gpu_capture.cpp` — 实现 stream/handle → client 映射；`orion_set_capture` C 导出
- `src/cuda_intercept.cpp` — 新增 `cudaLaunchKernelExC`, `cuLaunchKernel` wrapper 和对应 execute 函数；把 `get_current_client_idx` 换成 `resolve_client_idx`
- `src/cudnn_intercept.cpp` — 新增 `cudnnSetStream` 拦截；所有 wrapper 用 `resolve_client_idx`
- `src/cublas_intercept.cpp` — `cublasSetStream_v2` 补 handle 登记；wrapper 用 `resolve_client_idx`
- `src/scheduler.cpp` — 改 `start()` 为多 dispatcher；`dispatch_for_client`；GC init 后 dummy prime；`execute_kernel_launch_ex` / `_drv`
- `python/test_orion_blocking.py` — capture 时机；warmup 进 worker；stream 注册
- `Makefile` — 若新增 driver API 使用 `-ldl` 或链 `cuda`

### 可选文件
- `tests/` 下加一个单测：多 client 同时 launch → 全部落到对应 GC stream
- `docs/OPERATOR_LEVEL_SM_ALLOCATION_DETAILED_DESIGN.md` 更新对 "异步提交" 的假设

---

## 9. 开放问题（等决定）

1. **CUDA Graph 支持**：如果 PyTorch 启用了 `torch.cuda.CUDAGraph`，`cudaGraphLaunch` 会直接在 stream 上回放整张图，无法按 op 拆分拦截。本方案不覆盖这种场景，需要时另起方案（图内 kernel 不可隔离，只能整张图排队）。
2. **cublasLt JIT kernel**：部分 cuBLASLt kernel 是 JIT 编译后用 `cuLaunchKernel` 发射的；Step C 覆盖到，但 JIT 本身的耗时（几百 ms，首次）无法绕开，只能靠 Step A 的预热。
3. **多 GPU**：当前方案只考虑单 GPU；多 GPU 需要每 device 一套 GC context + dispatcher，`resolve_client_idx` 需要 `(device, stream) → client`。
4. **`cudaStreamSynchronize(NULL)`（legacy default stream）** 的语义：会等待设备所有流完成；在调度器接管后，这意味着要 drain 所有 client 的队列。实现时把它当作 `cudaDeviceSynchronize` 处理。

---

## 10. 实施顺序建议

1. **今天**：Step A（1-2 小时）+ Step B 的 stream→client（2-3 小时） → 跑一次 trace 看 M1/M2 标准。
2. **明天**：Step B 补 handle 绑定 + Step C（0.5-1 天）→ 跑 trace 看 M3 标准。
3. **后续**：Step D 调度器重构（1-2 天，需回归测试）→ M4 标准。

每步独立可回滚，失败成本低。
