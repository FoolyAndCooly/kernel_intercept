#!/usr/bin/env python3
"""
Orion 调度测试脚本（重构后版本）

测试 HP (High-Priority) 和多个 BE (Best-Effort) 客户端的并发执行。

=== Orion 调度核心概念 ===
1. HP (High-Priority): 高优先级客户端，通常是延迟敏感的推理任务
2. BE (Best-Effort): 尽力而为客户端，通常是训练任务，可以被延迟
3. SM (Streaming Multiprocessor): GPU 的计算单元，Orion 通过控制 SM 分配来调度

=== 调度策略（简化版）===
- HP kernel 直接执行
- BE kernel 在 HP 队列为空时执行
- 内存操作直接执行

使用方法：
    # 1 HP + 1 BE (默认)
    LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py

    # 1 HP + 2 BE
    LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py --num-be 2

    # 1 HP + 3 BE, 4 iterations
    LD_PRELOAD=./build/libgpu_scheduler.so python3 python/test_orion_blocking.py --num-be 3 --num-iters 4
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import ctypes
import sys
import threading
import time
import os
import json
import argparse

sys.path.insert(0, ".")
import GPT as gpt_module
import LLaMA as llama_module
from VGG16 import VGG16Model

# ============================================================================
# 模型配置（与 profile_gpt_vgg16.sh 保持一致）
# ============================================================================
# GPT 配置 - 调小以减少 SM 占用
EMB_SIZE = 768      # 从 2048 减到 768
HEAD_SIZE = 64      # 从 512 减到 64
N_LAYER = 1
SEQUENCE_LEN = 128  # 从 1024 减到 128
GPT_BATCH_SIZE = 2  # 从 16 减到 2
VOCAB_SIZE = 10000

gpt_module.emb_size = EMB_SIZE
gpt_module.head_size = HEAD_SIZE
gpt_module.n_layer = N_LAYER
gpt_module.sequence_len = SEQUENCE_LEN

from GPT import CharGPT
from LLaMA import CharLLaMA

# VGG 配置 - 调小以减少 SM 占用
VGG_BATCH_SIZE = 2   # 从 8 减到 2
VGG_IMAGE_SIZE = 112 # 从 224 减到 112


def save_results_tsv(result, num_be, output_file=None, project_root=None):
    """保存结果到TSV文件"""
    if output_file is None:
        if project_root:
            output_file = os.path.join(project_root, f"profiles/orion_1hp_{num_be}be_results.tsv")
        else:
            output_file = f"profiles/orion_1hp_{num_be}be_results.tsv"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("method\thp_time_ms\ttotal_time_ms\n")
        f.write(f"orion_1hp_{num_be}be\t{result['hp']:.2f}\t{result['total']:.2f}\n")
    print(f"Results saved to {output_file}")


def add_thread_names_to_trace(trace_file, thread_ids, num_clients):
    """后处理 Chrome trace 文件，添加线程名称标记"""
    with open(trace_file, 'r') as f:
        trace = json.load(f)

    events = trace.get('traceEvents', [])
    if not events:
        return

    pid = None
    for e in events:
        if e.get('pid') and isinstance(e.get('pid'), int):
            pid = e.get('pid')
            break
    if pid is None:
        return

    scheduler_tids = {}
    client_tid_set = set(thread_ids.values())
    for e in events:
        tid = e.get('tid')
        cat = e.get('cat', '')
        if tid and isinstance(tid, int) and tid > 100000 and tid not in client_tid_set:
            if 'cuda_runtime' in cat and tid not in scheduler_tids:
                scheduler_tids[tid] = len(scheduler_tids)

    stream_tids = {}
    for e in events:
        tid = e.get('tid')
        cat = e.get('cat', '')
        if tid and isinstance(tid, int) and tid < 100 and 'kernel' in cat:
            if tid not in stream_tids:
                stream_tids[tid] = len(stream_tids)

    tids_to_rename = set(thread_ids.values()) | set(scheduler_tids.keys()) | set(stream_tids.keys())
    new_events = []
    for e in events:
        if e.get('name') == 'thread_name' and e.get('tid') in tids_to_rename:
            continue
        new_events.append(e)

    for client_idx, tid in thread_ids.items():
        name = "HP-Client" if client_idx == 0 else f"BE{client_idx}-Client"
        new_events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": name}})

    for i, sched_tid in enumerate(sorted(scheduler_tids.keys())):
        name = "Scheduler"
        new_events.append({"ph": "M", "pid": pid, "tid": sched_tid, "name": "thread_name", "args": {"name": name}})

    for i, stream_tid in enumerate(sorted(stream_tids.keys())):
        if i == 0:
            name = f"GPU-Stream-HP(tid={stream_tid})"
        else:
            name = f"GPU-Stream-BE{i}(tid={stream_tid})"
        new_events.append({"ph": "M", "pid": 0, "tid": stream_tid, "name": "thread_name", "args": {"name": name}})

    trace['traceEvents'] = new_events
    with open(trace_file, 'w') as f:
        json.dump(trace, f)


def load_library():
    """加载 Orion 调度器的共享库"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    lib_path = os.path.join(project_root, "build", "libgpu_scheduler.so")

    if not os.path.exists(lib_path):
        print(f"ERROR: Library not found at {lib_path}")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

    lib.orion_init_scheduler.argtypes = [ctypes.c_int]
    lib.orion_init_scheduler.restype = ctypes.c_int
    lib.orion_start_scheduler_thread.argtypes = []
    lib.orion_start_scheduler_thread.restype = ctypes.c_int
    lib.orion_start_scheduler.argtypes = [ctypes.c_int]
    lib.orion_start_scheduler.restype = ctypes.c_int
    lib.orion_stop_scheduler.restype = None
    lib.orion_set_client_idx.argtypes = [ctypes.c_int]
    lib.orion_set_client_idx.restype = None
    lib.orion_load_kernel_info.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.orion_load_kernel_info.restype = ctypes.c_int
    lib.orion_set_client_kernels.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.orion_set_client_kernels.restype = None
    lib.orion_set_sm_threshold.argtypes = [ctypes.c_int]
    lib.orion_set_sm_threshold.restype = None
    lib.orion_get_sm_threshold.argtypes = []
    lib.orion_get_sm_threshold.restype = ctypes.c_int
    lib.orion_sync_client_stream.argtypes = [ctypes.c_int]
    lib.orion_sync_client_stream.restype = None
    lib.orion_reset_state.restype = None
    lib.orion_autotune_green_ctx.argtypes = []
    lib.orion_autotune_green_ctx.restype = ctypes.c_int

    # Step A/B: capture 开关 + stream→client 注册
    lib.orion_set_capture.argtypes = [ctypes.c_int]
    lib.orion_set_capture.restype = ctypes.c_int
    lib.orion_set_enabled.argtypes = [ctypes.c_int]
    lib.orion_set_enabled.restype = None
    lib.orion_register_client_stream.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.orion_register_client_stream.restype = None
    lib.orion_register_client_handle.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.orion_register_client_handle.restype = None

    print("Library loaded successfully")
    return lib


def run_test(lib, num_be, num_iters, kernel_info_paths, trace_file, sm_threshold=None, **kwargs):
    """
    运行 Orion 调度测试

    Args:
        lib: 调度器共享库
        num_be: BE 任务数量
        num_iters: 每个任务的迭代次数
        kernel_info_paths: kernel profile CSV 文件路径列表 [hp_path, be_path, ...]
        trace_file: 输出的 Chrome trace 文件路径
        sm_threshold: SM 阈值
    """
    num_clients = 1 + num_be  # 1 HP + num_be BE

    print(f"\n{'='*60}")
    print(f"Orion Scheduling Test")
    print(f"{'='*60}")
    print(f"Configuration: 1 HP + {num_be} BE, {num_iters} iterations each")

    # 第一步：初始化调度器（不启动线程）
    ret = lib.orion_init_scheduler(num_clients)
    if ret != 0:
        print("ERROR: Failed to initialize scheduler")
        return None

    # Step A：模型构造 / kernel profile 加载期间先关闭 capture，避免把一次性初始化
    # 操作（module 拷贝、cuDNN handle 首次绑定等）当成真正的 workload 入队。
    lib.orion_set_capture(0)

    if sm_threshold is not None:
        lib.orion_set_sm_threshold(sm_threshold)
    current_threshold = lib.orion_get_sm_threshold()
    print(f"SM threshold: {current_threshold}")

    # 创建模型和输入数据
    print("\nCreating models...")

    # HP (client 0): GPT
    gpt_model = CharGPT(vs=VOCAB_SIZE).to("cuda").eval()
    gpt_params = sum(p.numel() for p in gpt_model.parameters())
    print(f"  [HP] GPT params: {gpt_params/1e6:.1f}M")
    print(f"       Config: emb={EMB_SIZE}, heads={HEAD_SIZE}, layers={N_LAYER}")
    print(f"       Input: batch={GPT_BATCH_SIZE}, seq_len={SEQUENCE_LEN}")

    # BE (client 1+): 根据 be_model 参数选择模型
    models = [gpt_model]
    inputs = [torch.randint(0, VOCAB_SIZE, (GPT_BATCH_SIZE, SEQUENCE_LEN), device="cuda")]

    be_model_type = kwargs.get('be_model', 'gpt')

    for i in range(num_be):
        if be_model_type == 'llama':
            llama_module.hidden_size = EMB_SIZE
            llama_module.num_heads = EMB_SIZE // HEAD_SIZE
            llama_module.num_kv_heads = EMB_SIZE // HEAD_SIZE
            llama_module.num_layers = N_LAYER
            llama_module.intermediate_size = EMB_SIZE * 4
            llama_module.sequence_len = SEQUENCE_LEN
            be_model = CharLLaMA(vs=VOCAB_SIZE).to("cuda").eval()
            be_params = sum(p.numel() for p in be_model.parameters())
            print(f"  [BE{i+1}] LLaMA params: {be_params/1e6:.1f}M")
            print(f"        Config: hidden={EMB_SIZE}, heads={EMB_SIZE//HEAD_SIZE}, layers={N_LAYER}")
            print(f"        Input: batch={GPT_BATCH_SIZE}, seq_len={SEQUENCE_LEN}")
            models.append(be_model)
            inputs.append(torch.randint(0, VOCAB_SIZE, (GPT_BATCH_SIZE, SEQUENCE_LEN), device="cuda"))
        else:
            be_gpt_model = CharGPT(vs=VOCAB_SIZE).to("cuda").eval()
            be_gpt_params = sum(p.numel() for p in be_gpt_model.parameters())
            print(f"  [BE{i+1}] GPT params: {be_gpt_params/1e6:.1f}M")
            print(f"        Config: emb={EMB_SIZE}, heads={HEAD_SIZE}, layers={N_LAYER}")
            print(f"        Input: batch={GPT_BATCH_SIZE}, seq_len={SEQUENCE_LEN}")
            models.append(be_gpt_model)
            inputs.append(torch.randint(0, VOCAB_SIZE, (GPT_BATCH_SIZE, SEQUENCE_LEN), device="cuda"))

    torch.cuda.synchronize()

    # ========================================================================
    # 串行 Warmup（在 Green Context 创建之前执行）
    #
    # 测试：使用两个 GPT（都是 cuBLAS）代替 GPT+VGG16，验证是否是
    # cuDNN 导致的卡死问题。
    # ========================================================================
    print("\nSerial warmup (before Green Context creation)...")
    for idx in range(num_clients):
        client_type = "HP" if idx == 0 else f"BE{idx}"
        print(f"  Warming up {client_type} (client {idx})...")
        with torch.no_grad():
            _ = models[idx](inputs[idx])
        torch.cuda.synchronize()
        print(f"  {client_type} warmup done")

    # 第二步：加载 kernel profile 并 autotune（warmup 之后）
    if kernel_info_paths and len(kernel_info_paths) >= num_clients:
        print(f"\nLoading kernel profiles...")
        for i in range(num_clients):
            if kernel_info_paths[i] and os.path.exists(kernel_info_paths[i]):
                ret = lib.orion_load_kernel_info(i, kernel_info_paths[i].encode())
                print(f"  Client {i}: loaded {ret} kernel profiles from {os.path.basename(kernel_info_paths[i])}")
                lib.orion_set_client_kernels(i, ret if ret > 0 else 100)
            else:
                print(f"  Client {i}: no profile found at {kernel_info_paths[i]}")

        if os.getenv("ORION_GC_AUTOTUNE") == "1":
            print("\nAuto-tuning Green Context SM allocation...")
            ret = lib.orion_autotune_green_ctx()
            if ret == 0:
                print("  Green Context auto-tune successful")
            else:
                print(f"  Green Context auto-tune failed (ret={ret})")
    else:
        print("\nNo kernel profile files, using default scheduling")

    # Green Context 模式下，Orion 内部已经创建了 stream，不应该再创建外部 stream
    use_green_context = os.getenv("ORION_GC_AUTOTUNE") == "1"

    if use_green_context:
        streams = [None] * num_clients
        print("Green Context mode: using Orion internal streams")
    else:
        streams = [torch.cuda.Stream() for _ in range(num_clients)]
        for i, s in enumerate(streams):
            lib.orion_register_client_stream(i, ctypes.c_void_p(s.cuda_stream))
        print("DEFAULT mode: using PyTorch streams")

    barrier = threading.Barrier(num_clients)
    start = threading.Event()
    done = [threading.Event() for _ in range(num_clients)]
    client_times = [0.0] * num_clients
    client_start_times = [0.0] * num_clients
    client_end_times = [0.0] * num_clients
    thread_ids = {}

    def worker(idx):
        """客户端工作线程（仅执行，warmup 已在主线程完成）"""
        libc = ctypes.CDLL('libc.so.6')
        SYS_gettid = 186
        tid = libc.syscall(SYS_gettid)
        thread_ids[idx] = tid

        lib.orion_set_client_idx(idx)
        client_type = "HP" if idx == 0 else f"BE{idx}"

        # 注册当前线程的 PyTorch default stream 到 client 映射表，
        # 让 PyTorch 内部在该线程上用 default stream 发起的操作也能被正确路由
        cur_stream = torch.cuda.current_stream()
        lib.orion_register_client_stream(idx, ctypes.c_void_p(cur_stream.cuda_stream))

        # 等待所有线程同时开始
        start.wait()
        barrier.wait()

        # 开始计时
        t0 = time.time()
        client_start_times[idx] = t0

        # 执行模型推理
        if streams[idx] is not None:
            with torch.cuda.stream(streams[idx]):
                with torch.no_grad():
                    for i in range(num_iters):
                        _ = models[idx](inputs[idx])
            lib.orion_sync_client_stream(idx)
        else:
            with torch.no_grad():
                for i in range(num_iters):
                    _ = models[idx](inputs[idx])
            torch.cuda.synchronize()

        # 结束计时
        t1 = time.time()
        client_end_times[idx] = t1
        client_times[idx] = (t1 - t0) * 1000

        print(f"  {client_type}: {num_iters} iters in {client_times[idx]:.2f} ms")
        done[idx].set()

    # 创建并启动工作线程
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_clients)]
    for t in threads:
        t.start()

    # 启动调度器线程并打开 capture（warmup 已完成，不会有空转）
    lib.orion_start_scheduler_thread()
    print(f"Scheduler started with {num_clients} clients")
    lib.orion_set_capture(1)
    print("Capture enabled")

    os.makedirs(os.path.dirname(trace_file) if os.path.dirname(trace_file) else ".", exist_ok=True)

    print(f"\nRunning {num_iters} iterations per client...")

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        start.set()

        for d in done:
            d.wait()

        for t in threads:
            t.join()

        torch.cuda.synchronize()
        time.sleep(0.3)

    # 计算总时间：所有流上第一个算子开始到最后一个算子结束
    first_start = min(client_start_times)
    last_end = max(client_end_times)
    total_time = (last_end - first_start) * 1000

    prof.export_chrome_trace(trace_file)
    add_thread_names_to_trace(trace_file, thread_ids, num_clients)
    print(f"\nTrace saved to {trace_file}")

    # 输出结果
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"HP (client 0): {client_times[0]:.2f} ms")
    for i in range(1, num_clients):
        print(f"BE{i} (client {i}): {client_times[i]:.2f} ms")

    lib.orion_stop_scheduler()

    return {
        'total': total_time,
        'hp': client_times[0],
        'be': client_times[1:],
        'sm_threshold': current_threshold
    }


def main():
    parser = argparse.ArgumentParser(description='Orion Scheduling Test')
    parser.add_argument('--num-be', type=int, default=1, help='Number of BE tasks (default: 1)')
    parser.add_argument('--num-iters', type=int, default=2, help='Iterations per task (default: 2)')
    parser.add_argument('--sm-threshold', type=int, default=30, help='SM threshold (default: 30)')
    parser.add_argument('--output', type=str, default=None, help='Output trace file path')
    parser.add_argument('--be-model', type=str, default='gpt', choices=['gpt', 'llama'],
                        help='Model type for BE clients (default: gpt)')
    args = parser.parse_args()

    lib = load_library()

    # 使用相对路径，自动适配当前项目位置
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 查找最新的 profile 目录（profile_YYYYMMDD_HHMMSS）
    profiles_dir = os.path.join(project_root, "profiles")
    latest_profile_dir = None
    if os.path.exists(profiles_dir):
        profile_dirs = [d for d in os.listdir(profiles_dir)
                       if d.startswith("profile_") and os.path.isdir(os.path.join(profiles_dir, d))]
        if profile_dirs:
            latest_profile_dir = os.path.join(profiles_dir, sorted(profile_dirs)[-1])

    # 构建 kernel_info 路径列表：[hp_path, be1_path, be2_path, ...]
    kernel_info_list = []
    if latest_profile_dir:
        # HP (client 0): GPT
        gpt_profile = os.path.join(latest_profile_dir, "gpt", "kernel_info.csv")
        kernel_info_list.append(gpt_profile if os.path.exists(gpt_profile) else None)

        # BE (client 1+): VGG16
        for i in range(args.num_be):
            vgg_profile = os.path.join(latest_profile_dir, "vgg16", "kernel_info.csv")
            kernel_info_list.append(vgg_profile if os.path.exists(vgg_profile) else None)

        print(f"Using profile directory: {latest_profile_dir}")
    else:
        print("No profile directory found, using default scheduling")
        kernel_info_list = [None] * (1 + args.num_be)

    # 使用项目根目录的 profiles 目录
    trace_file = args.output or os.path.join(project_root, f"profiles/orion_gpt_vgg_{args.num_be}be_trace.json")
    result = run_test(lib, args.num_be, args.num_iters, kernel_info_list, trace_file, args.sm_threshold, be_model=args.be_model)

    if result:
        print(f"\n{'='*60}")
        print("TEST COMPLETE")
        print(f"{'='*60}")
        print(f"\nView trace in Chrome: chrome://tracing -> Load {trace_file}")

        # 保存TSV结果
        save_results_tsv(result, args.num_be, project_root=project_root)


if __name__ == "__main__":
    main()
