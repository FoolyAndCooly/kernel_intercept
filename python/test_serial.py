#!/usr/bin/env python3
"""
串行执行测试脚本

在同一个流上串行运行多个任务（HP 和 BE 顺序执行）。
用于与调度版本和并行版本对比。

使用方法：
    # 1 HP + 1 BE (默认)
    python3 python/test_serial.py

    # 1 HP + 2 BE
    python3 python/test_serial.py --num-be 2

    # 1 HP + 3 BE, 4 iterations
    python3 python/test_serial.py --num-be 3 --num-iters 4
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import sys
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
        output_file = f"profiles/serial_1hp_{num_be}be_results.tsv"
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("method\thp_time_ms\ttotal_time_ms\n")
        f.write(f"serial_1hp_{num_be}be\t{result['hp']:.2f}\t{result['total']:.2f}\n")
    print(f"Results saved to {output_file}")


def run_test(num_be, num_iters, trace_file):
    """运行串行测试"""
    num_tasks = 1 + num_be  # 1 HP + num_be BE

    print(f"\n{'='*60}")
    print("Serial Execution Test (Single Stream)")
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

    # 创建输出目录
    os.makedirs(os.path.dirname(trace_file) if os.path.dirname(trace_file) else ".", exist_ok=True)

    # 运行测试
    print(f"\nRunning {num_iters} iterations per task (serial)...")

    task_times = []

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        total_start = time.time()

        # 串行执行：Task 0 (HP) 然后 Task 1+ (BE)
        for task_idx in range(num_tasks):
            task_name = "HP" if task_idx == 0 else f"BE{task_idx}"
            task_start = time.time()

            with torch.no_grad():
                for i in range(num_iters):
                    _ = model(inputs[task_idx])

            torch.cuda.synchronize()
            task_time = (time.time() - task_start) * 1000
            task_times.append(task_time)
            print(f"  {task_name}: {num_iters} iters in {task_time:.2f} ms")

        total_time = (time.time() - total_start) * 1000

    # 导出 trace
    prof.export_chrome_trace(trace_file)
    print(f"\nTrace saved to {trace_file}")

    # 输出结果
    print(f"\n{'='*60}")
    print("Results (Serial)")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f} ms")
    print(f"HP (task 0): {task_times[0]:.2f} ms")
    for i in range(1, num_tasks):
        print(f"BE{i} (task {i}): {task_times[i]:.2f} ms")

    return {
        'total': total_time,
        'hp': task_times[0],
        'tasks': task_times
    }


def main():
    parser = argparse.ArgumentParser(description='Serial Execution Test')
    parser.add_argument('--num-be', type=int, default=1, help='Number of BE tasks (default: 1)')
    parser.add_argument('--num-iters', type=int, default=2, help='Iterations per task (default: 2)')
    parser.add_argument('--output', type=str, default=None, help='Output trace file path')
    args = parser.parse_args()

    trace_file = args.output or f"profiles/serial_1hp_{args.num_be}be_trace.json"
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
