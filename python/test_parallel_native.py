#!/usr/bin/env python3
"""
并行执行测试脚本（无调度）

使用多个独立的 CUDA stream 并行运行任务，不经过任何调度。
用于与调度版本和串行版本对比。

使用方法：
    # 1 HP + 1 BE (默认)
    python3 python/test_parallel_native.py

    # 1 HP + 2 BE
    python3 python/test_parallel_native.py --num-be 2

    # 1 HP + 3 BE, 4 iterations
    python3 python/test_parallel_native.py --num-be 3 --num-iters 4
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import sys
import threading
import time
import os
import argparse

sys.path.insert(0, ".")
import GPT as gpt_module

# ============================================================================
# 模型配置（与 test_orion_blocking.py 一致）
# ============================================================================
EMB_SIZE = 512
HEAD_SIZE = 8
N_LAYER = 1
SEQUENCE_LEN = 1024
BATCH_SIZE = 8
VOCAB_SIZE = 65

gpt_module.emb_size = EMB_SIZE
gpt_module.head_size = HEAD_SIZE
gpt_module.n_layer = N_LAYER
gpt_module.sequence_len = SEQUENCE_LEN

from GPT import CharGPT


def save_results_tsv(result, num_be, output_file=None):
    """保存结果到TSV文件"""
    if output_file is None:
        output_file = f"profiles/parallel_native_1hp_{num_be}be_results.tsv"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("method\thp_time_ms\ttotal_time_ms\n")
        f.write(f"parallel_native_1hp_{num_be}be\t{result['tasks'][0]:.2f}\t{result['total']:.2f}\n")
    print(f"Results saved to {output_file}")


def run_test(num_be, num_iters, trace_file):
    """
    运行并行测试

    Args:
        num_be: BE 任务数量
        num_iters: 每个任务的迭代次数
        trace_file: 输出的 Chrome trace 文件路径
    """
    num_tasks = 1 + num_be  # 1 HP + num_be BE

    print(f"\n{'='*60}")
    print("Parallel Execution Test (Native Streams, No Scheduling)")
    print(f"{'='*60}")
    print(f"Configuration: 1 HP + {num_be} BE, {num_iters} iterations each")

    # 创建模型
    print("\nCreating model...")
    model = CharGPT(vs=VOCAB_SIZE).to("cuda")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params/1e6:.1f}M")
    print(f"  Config: emb={EMB_SIZE}, heads={HEAD_SIZE}, layers={N_LAYER}")
    print(f"  Input: batch={BATCH_SIZE}, seq_len={SEQUENCE_LEN}")

    # 创建输入数据
    inputs = [torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LEN), device="cuda")
              for _ in range(num_tasks)]
    torch.cuda.synchronize()

    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        _ = model(inputs[0])
    torch.cuda.synchronize()

    # 创建 CUDA streams
    streams = [torch.cuda.Stream() for _ in range(num_tasks)]

    # 同步原语
    barrier = threading.Barrier(num_tasks)
    start_event = threading.Event()
    done_events = [threading.Event() for _ in range(num_tasks)]
    task_times = [0.0] * num_tasks
    task_start_times = [0.0] * num_tasks
    task_end_times = [0.0] * num_tasks

    def worker(task_idx):
        """任务工作线程"""
        task_name = "HP" if task_idx == 0 else f"BE{task_idx}"

        # 等待开始信号
        start_event.wait()
        barrier.wait()

        task_start = time.time()
        task_start_times[task_idx] = task_start

        # 在独立的 stream 上执行
        with torch.cuda.stream(streams[task_idx]):
            with torch.no_grad():
                for i in range(num_iters):
                    _ = model(inputs[task_idx])

        streams[task_idx].synchronize()
        task_end = time.time()
        task_end_times[task_idx] = task_end
        task_times[task_idx] = (task_end - task_start) * 1000
        print(f"  {task_name}: {num_iters} iters in {task_times[task_idx]:.2f} ms")
        done_events[task_idx].set()

    # 创建并启动工作线程
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_tasks)]
    for t in threads:
        t.start()

    time.sleep(0.2)

    # 创建输出目录
    os.makedirs(os.path.dirname(trace_file) if os.path.dirname(trace_file) else ".", exist_ok=True)

    # 运行测试
    print(f"\nRunning {num_iters} iterations per task...")

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        # 触发所有线程开始
        start_event.set()

        # 等待所有线程完成
        for event in done_events:
            event.wait()

        for t in threads:
            t.join()

        torch.cuda.synchronize()
        time.sleep(0.3)

    # 计算总时间：所有流上第一个算子开始到最后一个算子结束
    first_start = min(task_start_times)
    last_end = max(task_end_times)
    total_time = (last_end - first_start) * 1000

    # 导出 trace
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to {trace_file}")

    # 输出结果
    print(f"\n{'='*60}")
    print("Results (Parallel Native)")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"HP (task 0): {task_times[0]:.2f} ms")
    for i in range(1, num_tasks):
        print(f"BE{i} (task {i}): {task_times[i]:.2f} ms")

    return {
        'total': total_time,
        'tasks': task_times
    }


def main():
    parser = argparse.ArgumentParser(description='Parallel Native Streams Test')
    parser.add_argument('--num-be', type=int, default=1, help='Number of BE tasks (default: 1)')
    parser.add_argument('--num-iters', type=int, default=2, help='Iterations per task (default: 2)')
    parser.add_argument('--output', type=str, default=None, help='Output trace file path')
    args = parser.parse_args()

    trace_file = args.output or f"profiles/parallel_native_1hp_{args.num_be}be_trace.json"
    result = run_test(args.num_be, args.num_iters, trace_file)

    if result:
        print(f"\n{'='*60}")
        print("TEST COMPLETE")
        print(f"{'='*60}")
        print(f"\nView trace in Chrome: chrome://tracing -> Load {trace_file}")

        # 保存TSV结果
        save_results_tsv(result, args.num_be)

    return result


if __name__ == "__main__":
    main()
