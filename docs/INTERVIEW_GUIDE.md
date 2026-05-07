# GPU 算子拦截与调度系统 - 面试完全指南

> 本文档详细解释项目的实现原理，帮助你在面试中全面回答任何相关问题。

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [核心技术：LD_PRELOAD 拦截机制](#3-核心技术ld_preload-拦截机制)
4. [核心数据结构详解](#4-核心数据结构详解)
5. [拦截层实现详解](#5-拦截层实现详解)
6. [调度器实现详解](#6-调度器实现详解)
7. [多客户端并发支持](#7-多客户端并发支持)
8. [并发测试功能](#8-并发测试功能)
9. [面试常见问题与答案](#9-面试常见问题与答案)

---

## 1. 项目概述

### 1.1 项目是什么？

这是一个 **GPU 算子拦截与调度系统**，类似于 Orion 论文的实现。它的核心功能是：

1. **拦截** 应用程序的所有 CUDA/cuBLAS/cuDNN API 调用
2. **捕获** 这些调用的参数，放入队列
3. **调度** 由专门的调度器线程决定何时、在哪个 stream 上执行这些操作
4. **支持多客户端** 多个 GPU 任务可以共享 GPU，调度器负责协调

### 1.2 为什么需要这个系统？

**问题背景**：在 GPU 集群中，多个任务共享 GPU 时会产生干扰：
- **高优先级任务 (HP)**: 如在线推理服务，对延迟敏感
- **低优先级任务 (BE)**: 如离线训练任务，对吞吐量敏感

如果让它们自由竞争 GPU 资源，HP 任务的延迟会大幅增加。

**解决方案**：通过拦截和调度，我们可以：
- 控制 kernel 的执行顺序
- 让 HP 任务优先执行
- 在 HP 空闲时让 BE 任务执行
- 实现 compute-bound 和 memory-bound kernel 的互补调度

### 1.3 项目文件结构

```
kernel_intercept/
├── include/                    # 头文件
│   ├── common.h               # 公共定义（操作类型、日志等）
│   ├── gpu_capture.h          # 拦截层接口和数据结构
│   ├── scheduler.h            # 调度器接口
│   └── kernel_profile.h       # Kernel profile 信息
├── src/                       # 源文件
│   ├── gpu_capture.cpp        # 拦截层核心实现
│   ├── cuda_intercept.cpp     # CUDA Runtime API 拦截
│   ├── cublas_intercept.cpp   # cuBLAS API 拦截
│   ├── cudnn_intercept.cpp    # cuDNN API 拦截
│   └── scheduler.cpp          # 调度器实现
├── python/                    # Python 测试脚本
│   └── multiclient_intercept.py  # 多客户端测试
├── Makefile                   # 构建脚本
└── build/
    └── libgpu_scheduler.so    # 编译产物（共享库）
```

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        应用程序 (PyTorch)                            │
│   Client 0 (HP)          Client 1 (BE)          Client 2 (BE)       │
│   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐     │
│   │ model(x)    │        │ model(x)    │        │ model(x)    │     │
│   └──────┬──────┘        └──────┬──────┘        └──────┬──────┘     │
└──────────┼──────────────────────┼──────────────────────┼────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    拦截层 (LD_PRELOAD)                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  cudaLaunchKernel() / cublasSgemm() / cudnnConvolutionForward()│  │
│  │                           ↓                                     │  │
│  │  1. 检查是否需要拦截 (tl_is_scheduler_thread?)                  │  │
│  │  2. 创建 OperationRecord，保存所有参数                          │  │
│  │  3. 放入对应 client 的队列                                      │  │
│  │  4. 等待调度器执行完成                                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Per-Client 队列                                  │
│   Queue[0]               Queue[1]               Queue[2]             │
│   ┌─────────┐            ┌─────────┐            ┌─────────┐          │
│   │ Op1     │            │ Op1     │            │ Op1     │          │
│   │ Op2     │            │ Op2     │            │ Op2     │          │
│   │ ...     │            │ ...     │            │ ...     │          │
│   └─────────┘            └─────────┘            └─────────┘          │
└──────────────────────────────────────────────────────────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         调度器线程                                    │
│   Thread[0] (HP)         Thread[1] (BE)         Thread[2] (BE)       │
│   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐      │
│   │ 从 Queue[0] │        │ 从 Queue[1] │        │ 从 Queue[2] │      │
│   │ 取操作执行   │        │ 取操作执行   │        │ 取操作执行   │      │
│   │ 使用 HP     │        │ 使用 BE     │        │ 使用 BE     │      │
│   │ Stream      │        │ Stream[0]   │        │ Stream[1]   │      │
│   └─────────────┘        └─────────────┘        └─────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           GPU                                         │
│   HP Stream (高优先级)    BE Stream 0           BE Stream 1          │
│   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐      │
│   │ Kernel A    │        │ Kernel X    │        │ Kernel Y    │      │
│   │ Kernel B    │        │ Kernel Z    │        │ Kernel W    │      │
│   └─────────────┘        └─────────────┘        └─────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流详解

**以 `cublasSgemm` 为例，完整的数据流如下：**

```
1. PyTorch 调用 torch.mm(A, B)
      ↓
2. PyTorch 内部调用 cublasSgemm_v2()
      ↓
3. 由于 LD_PRELOAD，调用被重定向到我们的 cublasSgemm_v2() wrapper
      ↓
4. Wrapper 检查：
   - g_capture_state.initialized? (调度器是否初始化)
   - tl_is_scheduler_thread? (当前是否是调度器线程)
   - is_capture_enabled()? (拦截是否启用)
   - get_current_client_idx() >= 0? (是否是被管理的客户端线程)
      ↓
5. 如果需要拦截：
   a. 创建 OperationRecord
   b. 设置 params = CublasGemmParams{handle, m, n, k, alpha, A, B, beta, C, ...}
   c. 调用 enqueue_operation(op) 放入队列
   d. 调用 wait_operation(op) 等待完成
      ↓
6. 调度器线程从队列取出操作：
   a. 设置 cuBLAS handle 的 stream 为调度器的 stream
   b. 调用真实的 cublasSgemm_v2()
   c. 标记操作完成：op->mark_completed(result)
      ↓
7. 客户端线程被唤醒，返回结果
```

---

## 3. 核心技术：LD_PRELOAD 拦截机制

### 3.1 什么是 LD_PRELOAD？

`LD_PRELOAD` 是 Linux 动态链接器的一个环境变量，它允许你在程序启动时**优先加载指定的共享库**。

**原理**：
- 当程序调用一个函数（如 `cudaMalloc`）时，动态链接器会按顺序搜索共享库
- 如果 `LD_PRELOAD` 指定的库中有同名函数，就会优先使用它
- 这样我们就可以"劫持"原本的函数调用

**使用方式**：
```bash
LD_PRELOAD=./build/libgpu_scheduler.so python your_script.py
```

### 3.2 如何获取真实函数？

拦截后，我们还需要调用真实的 CUDA 函数。使用 `dlsym` 获取：

```cpp
// cuda_intercept.cpp:122-150
static void* get_cuda_func(const char* name) {
    if (!g_cudart_handle) {
        // 尝试多个可能的库路径
        const char* lib_paths[] = {
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.so",
            nullptr
        };

        for (int i = 0; lib_paths[i]; i++) {
            // RTLD_NOLOAD: 获取已加载的库句柄，不重新加载
            g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cudart_handle) break;
        }
        // ... 如果失败，尝试正常加载
    }

    if (g_cudart_handle) {
        void* fn = dlsym(g_cudart_handle, name);
        if (fn) return fn;
    }

    // 备选：使用 RTLD_DEFAULT 搜索所有已加载的库
    return dlsym(RTLD_DEFAULT, name);
}
```

**通俗解释**：
- `dlopen`: 打开一个共享库，获取句柄
- `dlsym`: 从共享库中查找函数地址
- `RTLD_NEXT`: 查找下一个同名符号（跳过当前库）
- `RTLD_DEFAULT`: 在所有已加载的库中搜索

### 3.3 拦截函数的基本模式

所有拦截函数都遵循相同的模式：

```cpp
// 以 cudaMalloc 为例 (cuda_intercept.cpp:410-431)
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    using namespace orion;

    // 1. 安全透传检查
    SAFE_PASSTHROUGH(cudaMalloc, devPtr, size);

    // 2. 检查拦截是否启用
    if (!is_capture_enabled()) {
        return real_cudaMalloc(devPtr, size);
    }

    // 3. 获取当前客户端索引
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudaMalloc(devPtr, size);  // 非管理线程，直接透传
    }

    // 4. 创建操作记录
    auto op = create_operation(client_idx, OperationType::MALLOC);
    if (!op) return real_cudaMalloc(devPtr, size);

    // 5. 设置参数
    op->params = MallocParams{devPtr, size};

    // 6. 入队
    enqueue_operation(op);

    // 7. 等待完成
    wait_operation(op);

    // 8. 返回结果
    return op->result;
}
```

### 3.4 SAFE_PASSTHROUGH 宏详解

这是一个关键的宏，用于决定是否需要拦截：

```cpp
// cuda_intercept.cpp:185-194
#define SAFE_PASSTHROUGH(func_name, ...) \
    do { \
        // 条件1: 调度器未初始化
        // 条件2: 当前是调度器线程（避免递归拦截）
        if (!g_capture_state.initialized.load() || tl_is_scheduler_thread) { \
            if (!g_real_funcs.initialized) init_real_functions(); \
            if (g_real_funcs.func_name) { \
                return g_real_funcs.func_name(__VA_ARGS__); \
            } \
            return cudaErrorUnknown; \
        } \
    } while(0)
```

**为什么需要 `tl_is_scheduler_thread`？**

这是防止**递归拦截**的关键！

想象这个场景：
1. 客户端线程调用 `cublasSgemm`
2. 被拦截，放入队列
3. 调度器线程取出操作，调用真实的 `cublasSgemm`
4. cuBLAS 内部会调用 `cudaLaunchKernel`
5. 如果不检查，`cudaLaunchKernel` 又会被拦截，放入队列
6. 调度器线程又要处理这个操作... **死循环！**

通过 `tl_is_scheduler_thread`，调度器线程的所有 CUDA 调用都会直接透传，避免递归。

### 3.5 dlsym 拦截的特殊处理

cuBLAS 使用 `dlsym` 动态获取某些函数，绕过了 LD_PRELOAD。我们通过拦截 `dlsym` 本身来解决：

```cpp
// cuda_intercept.cpp:696-723
void* dlsym(void* handle, const char* symbol) {
    // 获取真实的 dlsym
    if (!real_dlsym) {
        real_dlsym = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    // 拦截 cudaStreamGetCaptureInfo_v2
    // cuBLAS 在每次操作前调用此函数检查是否在 graph capture 模式
    if (symbol && strcmp(symbol, "cudaStreamGetCaptureInfo_v2") == 0) {
        return (void*)cudaStreamGetCaptureInfo_v2;  // 返回我们的版本
    }

    // 其他符号使用真实的 dlsym
    return real_dlsym(handle, symbol);
}
```

**为什么要拦截这个？**
- cuBLAS 在每次 GEMM 操作前会检查 stream 是否在 CUDA Graph Capture 模式
- 它通过 `dlsym` 获取 `cudaStreamGetCaptureInfo_v2` 函数
- 我们返回一个总是说"不在 capture 模式"的版本，减少开销

---

## 4. 核心数据结构详解

### 4.1 OperationRecord - 操作记录

这是最核心的数据结构，代表一个被拦截的 CUDA 操作：

```cpp
// gpu_capture.h:209-276
struct OperationRecord {
    OperationType type;           // 操作类型（KERNEL_LAUNCH, MALLOC, CUBLAS_SGEMM 等）
    uint64_t op_id;               // 全局唯一操作 ID
    int client_idx;               // 所属客户端索引

    // 操作参数 - 使用 std::variant 存储不同类型的参数
    std::variant<
        KernelLaunchParams,       // cudaLaunchKernel 参数
        MallocParams,             // cudaMalloc 参数
        FreeParams,               // cudaFree 参数
        MemcpyParams,             // cudaMemcpy 参数
        MemsetParams,             // cudaMemset 参数
        SyncParams,               // 同步操作参数
        CudnnConvParams,          // cuDNN 卷积参数
        CudnnBatchNormParams,     // cuDNN BatchNorm 参数
        CublasGemmParams,         // cuBLAS GEMM 参数
        CublasLtMatmulParams      // cuBLASLt Matmul 参数
    > params;

    // Profiling 信息
    std::string kernel_id;
    float estimated_duration_ms;
    int sm_needed;
    ProfileType profile_type;     // COMPUTE_BOUND 或 MEMORY_BOUND

    // 执行状态
    std::atomic<bool> completed{false};
    std::atomic<bool> started{false};
    cudaError_t result;
    void* result_ptr;             // malloc 的返回指针

    // 同步机制
    std::mutex completion_mutex;
    std::condition_variable completion_cv;

    // 等待操作完成
    void wait_completion() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] {
            return completed.load(std::memory_order_acquire);
        });
    }

    // 标记操作完成
    void mark_completed(cudaError_t res) {
        {
            std::lock_guard<std::mutex> lock(completion_mutex);
            result = res;
            completed.store(true, std::memory_order_release);
        }
        completion_cv.notify_all();  // 唤醒等待的客户端线程
    }
};
```

**通俗解释**：
- `OperationRecord` 就像一个"工单"
- 客户端线程创建工单，填写参数，放入队列
- 调度器线程取出工单，执行操作，标记完成
- 客户端线程等待工单完成，获取结果

### 4.2 std::variant 的使用

`std::variant` 是 C++17 的类型安全联合体，可以存储多种类型中的一种：

```cpp
// 设置参数
op->params = CublasGemmParams{handle, m, n, k, ...};

// 获取参数（需要知道当前存储的类型）
auto& p = std::get<CublasGemmParams>(op->params);

// 检查当前存储的类型索引
if (op->params.index() == 0) {  // KernelLaunchParams
    auto& kp = std::get<KernelLaunchParams>(op->params);
}
```

### 4.3 KernelLaunchParams - Kernel 启动参数

```cpp
// gpu_capture.h:31-68
struct KernelLaunchParams {
    const void* func;             // kernel 函数指针
    dim3 gridDim;                 // grid 维度
    dim3 blockDim;                // block 维度
    size_t sharedMem;             // 共享内存大小
    cudaStream_t stream;          // 原始 stream（会被调度器替换）

    // 方案 A: 保存原始 args 指针（同步模式使用）
    void** original_args;

    // 方案 B: 深拷贝参数（异步模式使用）
    std::vector<uint8_t> args_buffer;
    std::vector<size_t> args_offsets;
    std::vector<size_t> args_sizes;
    size_t num_args;
    std::vector<void*> args_ptrs;
    bool use_deep_copy;

    // 获取用于执行的 args 指针
    void** get_args() {
        if (use_deep_copy && num_args > 0) {
            // 重建 args 指针数组
            args_ptrs.resize(num_args);
            for (size_t i = 0; i < num_args; i++) {
                args_ptrs[i] = args_buffer.data() + args_offsets[i];
            }
            return args_ptrs.data();
        }
        return original_args;
    }
};
```

**为什么需要两种方案？**

- **同步模式**：客户端线程等待操作完成，`original_args` 指向的内存一直有效
- **异步模式**：客户端线程不等待，`original_args` 可能已经失效，需要深拷贝

**问题**：对于 `cudaLaunchKernel`，我们无法知道每个参数的大小，所以目前只能用同步模式。
但对于 cuBLAS，参数结构是已知的，可以安全地深拷贝。

### 4.4 ClientQueue - 客户端队列

```cpp
// gpu_capture.h:284-343
class ClientQueue {
public:
    // 提交操作到队列
    void push(OperationPtr op) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(op));
        }
        cv_.notify_one();  // 唤醒等待的调度器线程
    }

    // 阻塞等待操作
    OperationPtr wait_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) return nullptr;
        OperationPtr op = std::move(queue_.front());
        queue_.pop();
        return op;
    }

    // 非阻塞尝试取出
    OperationPtr try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return nullptr;
        OperationPtr op = std::move(queue_.front());
        queue_.pop();
        return op;
    }

private:
    std::queue<OperationPtr> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_;
};
```

**设计要点**：
- 每个客户端有独立的队列，避免竞争
- 使用条件变量实现高效的等待/唤醒
- `shutdown_` 标志用于优雅关闭

### 4.5 CaptureLayerState - 全局拦截层状态

```cpp
// gpu_capture.h:349-382
struct CaptureLayerState {
    std::atomic<bool> initialized{false};  // 是否已初始化
    std::atomic<bool> enabled{false};      // 是否启用拦截

    int num_clients{0};                    // 客户端数量

    // Per-client 队列
    std::vector<std::unique_ptr<ClientQueue>> client_queues;

    // 操作 ID 生成器
    std::atomic<uint64_t> next_op_id{0};

    // Block/Unblock 同步原语
    std::atomic<bool>* client_blocked{nullptr};
    std::mutex* client_mutexes{nullptr};
    std::condition_variable* client_cvs{nullptr};

    // 调度器通知
    std::mutex scheduler_mutex;
    std::condition_variable scheduler_cv;

    std::atomic<bool> shutdown{false};
};

// 全局实例
extern CaptureLayerState g_capture_state;
```

### 4.6 线程局部变量

```cpp
// gpu_capture.cpp:17-24
// 当前线程的客户端索引（-1 表示非管理线程）
static thread_local int tl_client_idx = -1;

// 标识当前线程是否为调度器线程
thread_local bool tl_is_scheduler_thread = false;

// 重入保护标志
thread_local bool tl_in_scheduler_execution = false;
```

**`thread_local` 的作用**：
- 每个线程有自己独立的变量副本
- 客户端线程设置 `tl_client_idx` 为自己的索引
- 调度器线程设置 `tl_is_scheduler_thread = true`

---

## 5. 拦截层实现详解

### 5.1 初始化流程

```cpp
// gpu_capture.cpp:53-89
int init_capture_layer(int num_clients) {
    if (g_capture_state.initialized.load()) {
        return 0;  // 已初始化
    }

    if (num_clients <= 0 || num_clients > MAX_CLIENTS) {
        return -1;  // 参数错误
    }

    init_log_level();  // 从环境变量读取日志级别

    g_capture_state.num_clients = num_clients;

    // 初始化 per-client 队列
    g_capture_state.client_queues.resize(num_clients);
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_queues[i] = std::make_unique<ClientQueue>();
    }

    // 初始化同步原语
    g_capture_state.client_blocked = new std::atomic<bool>[num_clients];
    g_capture_state.client_mutexes = new std::mutex[num_clients];
    g_capture_state.client_cvs = new std::condition_variable[num_clients];

    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_blocked[i].store(false);
    }

    g_capture_state.shutdown.store(false);
    g_capture_state.initialized.store(true);
    g_capture_state.enabled.store(true);

    return 0;
}
```

### 5.2 操作创建和入队

```cpp
// gpu_capture.cpp:150-174
OperationPtr create_operation(int client_idx, OperationType type) {
    if (client_idx < 0 || client_idx >= g_capture_state.num_clients) {
        return nullptr;
    }

    auto op = std::make_shared<OperationRecord>();
    op->type = type;
    op->client_idx = client_idx;
    op->op_id = g_capture_state.next_op_id.fetch_add(1);  // 原子递增

    return op;
}

void enqueue_operation(OperationPtr op) {
    if (!op) return;

    // 放入对应客户端的队列
    g_capture_state.client_queues[op->client_idx]->push(op);

    // 通知调度器
    notify_scheduler();
}
```

### 5.3 cuBLAS 拦截的特殊处理

cuBLAS 拦截比 CUDA Runtime 更复杂，因为需要处理 stream 和 handle：

```cpp
// cublas_intercept.cpp:513-620 (简化版)
cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc) {

    using namespace orion;

    // 1. 重入保护
    if (tl_in_sgemm) {
        return g_cublas_funcs.cublasSgemm_v2(...);  // 直接透传
    }

    // 2. 各种检查（同 cudaMalloc）
    if (!g_capture_state.initialized.load()) return 透传;
    if (!is_capture_enabled()) return 透传;
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return 透传;

    // 3. 创建操作
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM);

    // 4. 检查异步模式
    bool async_mode = (get_async_mode_internal() == 1);

    // 5. 设置参数
    CublasGemmParams params;
    params.handle = handle;
    params.m = m; params.n = n; params.k = k;
    params.A = A; params.B = B; params.C = C;
    // ...

    if (async_mode) {
        // 异步模式：深拷贝 alpha 和 beta 的值
        params.alpha_value = *alpha;
        params.beta_value = *beta;
        params.use_stored_scalars = true;
    } else {
        // 同步模式：直接使用指针
        params.alpha = alpha;
        params.beta = beta;
    }

    op->params = params;
    enqueue_operation(op);

    if (async_mode) {
        return CUBLAS_STATUS_SUCCESS;  // 不等待
    }

    wait_operation(op);
    return op->result == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1;
}
```

**关键点**：
- **重入保护**：`tl_in_sgemm` 防止递归调用
- **异步模式**：深拷贝标量参数，因为客户端不等待，原指针可能失效
- **Stream 控制**：调度器执行时会设置自己的 stream

### 5.4 cublasSetStream 拦截

PyTorch 在每次 cuBLAS 操作前会设置 stream，我们需要拦截并忽略：

```cpp
// cublas_intercept.cpp:1183-1240
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId) {
    using namespace orion;

    // 调度器启用时，忽略 PyTorch 的 stream 设置
    // 让调度器在执行时设置自己的 stream
    if (g_capture_state.initialized.load() && is_capture_enabled()) {
        int client_idx = get_current_client_idx();
        if (client_idx >= 0) {
            // 清除 stream 绑定，让调度器控制
            return g_cublas_funcs.cublasSetStream_v2(handle, nullptr);
        }
    }

    // 否则正常透传
    return g_cublas_funcs.cublasSetStream_v2(handle, streamId);
}
```

---

## 6. 调度器实现详解

### 6.1 调度器配置

```cpp
// scheduler.h:20-60
struct SchedulerConfig {
    // SM 阈值: BE kernel 的最大 SM 占用比例
    float sm_threshold_ratio = 0.5f;

    // 时间阈值: BE kernel 累计执行时间占 HP 请求延迟的最大比例
    float dur_threshold_ratio = 0.025f;  // 2.5%

    // HP 请求的平均延迟 (ms)
    float hp_request_latency_ms = 10.0f;

    // 轮询间隔 (us)
    int poll_interval_us = 10;

    // 是否启用干扰感知调度
    bool interference_aware = true;

    // GPU SM 数量 (运行时获取)
    int num_sms = 0;

    // Compute/Memory 互补调度配置
    bool complementary_mode = true;
    int max_concurrent_compute = 4;
    int max_concurrent_memory = 4;

    // 计算实际阈值
    int get_sm_threshold() const {
        return static_cast<int>(num_sms * sm_threshold_ratio);
    }

    float get_dur_threshold_ms() const {
        return hp_request_latency_ms * dur_threshold_ratio;
    }
};
```

### 6.2 调度器初始化

```cpp
// scheduler.cpp:61-100
bool Scheduler::init(int num_clients, const SchedulerConfig& config) {
    if (initialized_.load()) return true;

    num_clients_ = num_clients;
    config_ = config;

    // 获取 GPU SM 数量
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config_.device_id);
    config_.num_sms = prop.multiProcessorCount;

    // 创建 CUDA streams
    if (!create_streams()) return false;

    initialized_.store(true);
    return true;
}

bool Scheduler::create_streams() {
    // 获取 stream 优先级范围
    int lowest_priority, highest_priority;
    cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);

    // 创建高优先级 stream (用于 HP 客户端)
    cudaStreamCreateWithPriority(&hp_stream_,
                                  cudaStreamNonBlocking,
                                  highest_priority);

    // 为每个 BE 客户端创建低优先级 stream
    be_streams_.resize(num_clients_ - 1);
    for (int i = 0; i < num_clients_ - 1; i++) {
        cudaStreamCreateWithPriority(&be_streams_[i],
                                      cudaStreamNonBlocking,
                                      lowest_priority);
    }
    return true;
}
```

**Stream 优先级**：
- CUDA 支持 stream 优先级，高优先级 stream 的 kernel 会优先调度
- HP 客户端使用最高优先级 stream
- BE 客户端使用最低优先级 stream

### 6.3 调度器线程启动

```cpp
// scheduler.cpp:146-164
void Scheduler::start() {
    if (!initialized_.load() || running_.load()) return;

    running_.store(true);

    // 为每个客户端启动一个调度器线程
    for (int i = 0; i < num_clients_; i++) {
        threads_.emplace_back(&Scheduler::run_client, this, i);
    }
}
```

**设计选择**：每个客户端一个调度器线程
- 优点：简化同步，每个线程只处理自己的队列
- 优点：天然支持并发，不同客户端的操作可以并行执行
- 缺点：线程数量随客户端增加

### 6.4 调度器线程主循环

```cpp
// scheduler.cpp:368-654 (简化版)
void Scheduler::run_client(int client_idx) {
    // 关键：标记当前线程为调度器线程
    tl_is_scheduler_thread = true;

    // 预初始化 cuBLAS handle
    preinit_thread_local_cublas_handle();

    // 选择使用的 stream
    cudaStream_t my_stream;
    if (client_idx == 0) {
        my_stream = hp_stream_;  // HP 使用高优先级 stream
    } else {
        my_stream = be_streams_[client_idx - 1];  // BE 使用低优先级 stream
    }

    while (running_.load()) {
        // 从对应的客户端队列取操作（阻塞等待）
        OperationPtr op = g_capture_state.client_queues[client_idx]->wait_pop();

        if (op) {
            // 执行操作
            cudaError_t err = execute_operation(op, my_stream);

            // 标记完成，唤醒客户端线程
            op->mark_completed(err);
        }
    }

    // 处理剩余操作
    while (!g_capture_state.client_queues[client_idx]->empty()) {
        auto op = g_capture_state.client_queues[client_idx]->try_pop();
        if (op) {
            cudaError_t err = execute_operation(op, my_stream);
            op->mark_completed(err);
        }
    }

    // 等待 stream 完成
    cudaStreamSynchronize(my_stream);
}
```

### 6.5 操作执行

```cpp
// scheduler.cpp:774-810
cudaError_t Scheduler::execute_operation(OperationPtr op, cudaStream_t stream) {
    op->started.store(true);

    switch (op->type) {
        case OperationType::KERNEL_LAUNCH:
        case OperationType::MALLOC:
        case OperationType::FREE:
        case OperationType::MEMCPY:
        case OperationType::MEMCPY_ASYNC:
        case OperationType::MEMSET:
        case OperationType::MEMSET_ASYNC:
        case OperationType::DEVICE_SYNC:
        case OperationType::STREAM_SYNC:
            return execute_cuda_operation(op, stream);

        case OperationType::CUDNN_CONV_FWD:
        case OperationType::CUDNN_CONV_BWD_DATA:
        case OperationType::CUDNN_CONV_BWD_FILTER:
        case OperationType::CUDNN_BATCHNORM_FWD:
        case OperationType::CUDNN_BATCHNORM_BWD:
            return execute_cudnn_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        case OperationType::CUBLAS_SGEMM:
        case OperationType::CUBLAS_SGEMM_BATCHED:
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
            return execute_cublas_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        case OperationType::CUBLASLT_MATMUL:
            return execute_cublaslt_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        default:
            return cudaErrorUnknown;
    }
}
```

### 6.6 HP/BE 调度决策（干扰感知）

```cpp
// scheduler.cpp:662-713
bool Scheduler::schedule_be(const OperationPtr& hp_op, const OperationPtr& be_op) {
    // 如果没有 HP 任务在执行，允许 BE
    if (!hp_task_running_.load()) {
        return true;
    }

    // 如果不启用干扰感知，允许并发
    if (!config_.interference_aware) {
        return true;
    }

    // 检查 BE 操作的 SM 需求
    int sm_needed = be_op ? be_op->sm_needed : config_.num_sms / 4;
    if (sm_needed >= config_.get_sm_threshold()) {
        return false;  // SM 需求太大，拒绝
    }

    // 检查 profile 类型是否互补
    ProfileType hp_type = hp_op ? hp_op->profile_type : ProfileType::UNKNOWN;
    ProfileType be_type = be_op ? be_op->profile_type : ProfileType::UNKNOWN;
    if (!is_complementary(hp_type, be_type)) {
        return false;  // 不互补，拒绝
    }

    // 检查累计 BE 时间是否超过阈值
    float be_duration = be_op ? be_op->estimated_duration_ms : 0.1f;
    if (cumulative_be_duration_ms_ + be_duration > config_.get_dur_threshold_ms()) {
        return false;  // 累计时间太长，拒绝
    }

    // 更新累计时间
    cumulative_be_duration_ms_ += be_duration;
    return true;
}

bool Scheduler::is_complementary(ProfileType hp_type, ProfileType be_type) {
    // 一个 compute-bound，一个 memory-bound 时互补
    return (hp_type == ProfileType::COMPUTE_BOUND && be_type == ProfileType::MEMORY_BOUND) ||
           (hp_type == ProfileType::MEMORY_BOUND && be_type == ProfileType::COMPUTE_BOUND);
}
```

**调度策略解释**：

1. **SM 阈值**：如果 BE kernel 需要太多 SM，会严重影响 HP，拒绝执行
2. **互补调度**：compute-bound 和 memory-bound kernel 可以并发，因为它们使用不同资源
3. **时间阈值**：限制 BE kernel 的累计执行时间，避免长时间影响 HP

---

## 7. 多客户端并发支持

### 7.1 线程本地 cuBLAS Handle

为了支持多个调度器线程并发执行 cuBLAS 操作，每个线程需要独立的 handle：

```cpp
// cublas_intercept.cpp:319-358
// 每个调度器线程的独立 cuBLAS handle
static thread_local cublasHandle_t tl_cublas_handle = nullptr;
static thread_local bool tl_cublas_handle_initialized = false;

static cublasHandle_t get_thread_local_cublas_handle(cudaStream_t stream) {
    if (!tl_cublas_handle_initialized && g_cublasCreate) {
        cublasStatus_t status = g_cublasCreate(&tl_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return nullptr;
        }
        tl_cublas_handle_initialized = true;
    }

    // 设置 stream
    if (tl_cublas_handle && stream && g_cublasSetStreamDirect) {
        g_cublasSetStreamDirect(tl_cublas_handle, stream);
    }

    return tl_cublas_handle;
}

// 预初始化（在调度器线程启动时调用）
void preinit_thread_local_cublas_handle() {
    get_thread_local_cublas_handle(nullptr);
}
```

**为什么需要线程本地 handle？**

如果多个调度器线程共享一个 cuBLAS handle：
1. 线程 A 设置 handle 的 stream 为 stream_A
2. 线程 B 设置 handle 的 stream 为 stream_B
3. 线程 A 执行 GEMM，但此时 stream 已经是 stream_B！

使用线程本地 handle 避免了这种竞争。

### 7.2 Python 端的使用方式

```python
# multiclient_intercept.py (简化版)
import ctypes
import threading

# 加载调度器库
lib = ctypes.CDLL("./build/libgpu_scheduler.so")

# 启动调度器（3 个客户端：1 HP + 2 BE）
lib.orion_start_scheduler(3)
lib.orion_set_async_mode(1)  # 启用异步模式

def client_worker(client_idx, model, input_data):
    # 设置当前线程的客户端索引
    lib.orion_set_client_idx(client_idx)

    # 执行模型推理
    with torch.no_grad():
        output = model(input_data)

    return output

# 创建线程
threads = []
for i in range(3):
    t = threading.Thread(target=client_worker, args=(i, model, inputs[i]))
    threads.append(t)

# 启动所有线程
for t in threads:
    t.start()

# 等待完成
for t in threads:
    t.join()

# 停止调度器
lib.orion_stop_scheduler()
```

---

## 8. 并发测试功能

### 8.1 功能概述

项目实现了一个并发测试功能，可以：
1. **收集模式**：记录所有操作的信息
2. **配对测试模式**：指定两个操作并发执行，测量干扰

### 8.2 测试状态结构

```cpp
// scheduler.h:287-359
struct ConcurrentTestState {
    ConcurrentTestMode mode{ConcurrentTestMode::DISABLED};

    // 收集模式：记录每个客户端的操作信息
    std::vector<std::vector<OpInfo>> client_ops;

    // 配对测试模式
    ConcurrentPair target_pair;

    // Stream kernel 计数器
    std::atomic<int> stream_kernel_count_a{0};
    std::atomic<int> stream_kernel_count_b{0};
    int target_kernel_idx_a{-1};
    int target_kernel_idx_b{-1};

    // 同步点
    std::mutex sync_mutex;
    std::condition_variable sync_cv;
    std::atomic<int> ops_at_sync_point{0};
    std::atomic<bool> pair_ready{false};

    // 存储到达同步点的操作
    OperationPtr op_a;
    OperationPtr op_b;
    cudaStream_t stream_a;
    cudaStream_t stream_b;
};
```

### 8.3 并发启动机制

```cpp
// scheduler.cpp:312-366
static cudaError_t launch_concurrent_kernel(bool is_a, OperationPtr op, cudaStream_t stream) {
    // 到达 barrier，等待另一个线程
    int count = g_launch_barrier.fetch_add(1) + 1;

    if (count == 2) {
        // 最后一个到达的线程发出启动信号
        g_launch_go.store(true);
    } else {
        // 等待启动信号（自旋等待，确保最小延迟）
        while (!g_launch_go.load()) {
            // 纯自旋，不 yield
        }
    }

    // 两个线程同时执行到这里，直接发 kernel
    cudaError_t err = execute_cuda_operation(op, stream);
    return err;
}
```

**设计要点**：
- 使用原子变量实现 barrier
- 自旋等待确保最小延迟
- 两个 kernel 几乎同时提交到 GPU

---

## 9. 面试常见问题与答案

### Q1: 项目的核心目标是什么？

**答**：实现一个 GPU 算子拦截与调度系统，用于：
1. 拦截应用程序的 CUDA/cuBLAS/cuDNN API 调用
2. 将操作放入队列，由调度器统一管理
3. 支持多客户端共享 GPU，实现 HP/BE 优先级调度
4. 通过互补调度（compute-bound + memory-bound）提高 GPU 利用率

### Q2: LD_PRELOAD 是如何工作的？

**答**：
1. `LD_PRELOAD` 是 Linux 动态链接器的环境变量
2. 它让指定的共享库优先于其他库加载
3. 如果我们的库中有同名函数（如 `cudaMalloc`），会优先使用我们的版本
4. 我们的函数可以做一些处理，然后调用真实的函数（通过 `dlsym` 获取）

### Q3: 如何防止递归拦截？

**答**：使用线程局部变量 `tl_is_scheduler_thread`：
1. 调度器线程启动时设置 `tl_is_scheduler_thread = true`
2. 拦截函数检查这个标志，如果是调度器线程就直接透传
3. 这样调度器执行 cuBLAS 时，内部的 `cudaLaunchKernel` 不会被再次拦截

### Q4: 为什么需要每个客户端一个队列？

**答**：
1. **避免竞争**：多个客户端同时提交操作时，不需要争抢同一个队列的锁
2. **保持顺序**：每个客户端的操作按提交顺序执行
3. **简化调度**：每个调度器线程只处理自己的队列

### Q5: 同步模式和异步模式有什么区别？

**答**：
- **同步模式**：客户端线程提交操作后等待完成，参数可以直接使用指针
- **异步模式**：客户端线程提交后立即返回，参数需要深拷贝（因为原指针可能失效）

对于 `cudaLaunchKernel`，由于无法知道参数大小，只能用同步模式。
对于 cuBLAS，参数结构已知，可以深拷贝 alpha/beta 值，支持异步模式。

### Q6: Stream 优先级是如何工作的？

**答**：
1. CUDA 支持 stream 优先级，通过 `cudaStreamCreateWithPriority` 创建
2. 高优先级 stream 的 kernel 会被 GPU 优先调度
3. HP 客户端使用最高优先级 stream，BE 客户端使用最低优先级
4. 这是硬件级别的优先级，比软件调度更有效

### Q7: 什么是互补调度？

**答**：
- **Compute-bound kernel**：主要使用计算单元（如 GEMM）
- **Memory-bound kernel**：主要使用内存带宽（如 BatchNorm）

当一个 compute-bound 和一个 memory-bound kernel 并发执行时，它们使用不同的资源，干扰较小。调度器会优先让这种互补的 kernel 并发执行。

### Q8: 如何处理 cuBLAS 的 stream 设置？

**答**：
1. PyTorch 在每次 cuBLAS 操作前会调用 `cublasSetStream_v2` 设置 stream
2. 我们拦截这个函数，当调度器启用时忽略 PyTorch 的设置
3. 调度器执行时使用自己的 stream
4. 每个调度器线程有独立的 cuBLAS handle，避免竞争

### Q9: 项目中用到了哪些 C++ 特性？

**答**：
1. **C++17 std::variant**：类型安全的联合体，存储不同类型的操作参数
2. **std::atomic**：原子操作，用于线程安全的状态管理
3. **std::condition_variable**：条件变量，用于线程间同步
4. **thread_local**：线程局部存储，每个线程有独立的变量副本
5. **std::shared_ptr**：智能指针，管理 OperationRecord 的生命周期
6. **Lambda 表达式**：用于条件变量的等待条件

### Q10: 如何保证线程安全？

**答**：
1. **原子变量**：`std::atomic` 用于简单的状态标志
2. **互斥锁**：`std::mutex` 保护共享数据结构
3. **条件变量**：`std::condition_variable` 用于等待/唤醒
4. **线程局部存储**：`thread_local` 避免共享
5. **每客户端独立队列**：减少竞争

### Q11: 项目的性能开销在哪里？

**答**：
1. **拦截开销**：每次 CUDA 调用都要检查是否需要拦截
2. **队列操作**：入队/出队需要加锁
3. **线程同步**：客户端等待调度器完成
4. **Stream 切换**：调度器使用自己的 stream

优化措施：
- 使用 `thread_local` 减少锁竞争
- 条件变量避免忙等待
- 预初始化 cuBLAS handle 避免延迟

### Q12: 如果让你改进这个项目，你会怎么做？

**答**：
1. **支持 cudaLaunchKernel 的异步模式**：通过 CUDA Driver API 获取参数大小
2. **更智能的调度策略**：基于历史数据预测 kernel 执行时间
3. **支持多 GPU**：扩展到多 GPU 场景
4. **减少拦截开销**：使用更高效的函数指针缓存
5. **添加监控和统计**：实时显示调度状态和性能指标

---

## 10. 总结

### 核心技术点

1. **LD_PRELOAD 拦截**：劫持 CUDA API 调用
2. **dlsym 获取真实函数**：调用原始实现
3. **线程局部变量**：区分客户端线程和调度器线程
4. **生产者-消费者模式**：客户端提交，调度器执行
5. **条件变量同步**：高效的等待/唤醒机制
6. **Stream 优先级**：硬件级别的优先级调度

### 关键设计决策

1. **每客户端一个队列**：避免竞争，保持顺序
2. **每客户端一个调度器线程**：简化同步，支持并发
3. **线程本地 cuBLAS handle**：避免 handle 竞争
4. **同步/异步模式**：平衡性能和正确性

### 面试时的表达建议

1. **先说整体架构**：拦截层 → 队列 → 调度器 → GPU
2. **再说关键技术**：LD_PRELOAD、dlsym、thread_local
3. **然后说设计决策**：为什么这样设计，有什么权衡
4. **最后说改进方向**：展示你的思考深度

祝面试顺利！
