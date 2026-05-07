# Orion GPU 算子拦截与调度系统

基于 LD_PRELOAD 机制的 CUDA 算子拦截和调度框架，支持 HP (High-Priority) 和 BE (Best-Effort) 多客户端并发调度。

### 1. 编译

```bash
make              # 构建 release 版本
make DEBUG=1      # 构建 debug 版本（更多日志）
```

生成文件：`build/libgpu_scheduler.so`

### 2. 运行测试

**使用启动脚本（推荐）：**

```bash
./run.sh                           # 默认 1 HP + 1 BE
./run.sh --num-be 2 --num-iters 4  # 1 HP + 2 BE, 4 次迭代
./run.sh --debug                   # 启用详细日志
```

**直接运行：**

```bash
LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py
```

### 3. 查看结果

运行后生成 Chrome trace 文件：

```bash
# 在 Chrome 浏览器中打开
chrome://tracing → Load → profiles/orion_1hp_1be_trace.json
```

## 项目结构

```
├── include/
│   ├── common.h          # 公共定义（操作类型、日志）
│   ├── gpu_capture.h     # 捕获层接口
│   ├── scheduler.h       # 调度器接口
│   └── kernel_profile.h  # Kernel 性能特征
├── src/
│   ├── cuda_intercept.cpp    # CUDA API 拦截
│   ├── cublas_intercept.cpp  # cuBLAS 拦截
│   ├── cudnn_intercept.cpp   # cuDNN 拦截
│   ├── gpu_capture.cpp       # 操作队列管理
│   └── scheduler.cpp         # 调度器实现
├── python/
│   ├── GPT.py                   # GPT 模型定义
│   ├── test_orion_blocking.py   # 调度测试脚本
│   ├── test_parallel_native.py  # 原生并行测试
│   └── launch_jobs.py           # 多模型启动器
├── profiles/                # 输出目录（trace、结果文件）
├── Makefile
└── run.sh                   # 便捷启动脚本
```
