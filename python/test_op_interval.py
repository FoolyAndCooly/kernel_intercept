#!/usr/bin/env python3
"""
测试算子间隔：单个 client 连续提交多个小 kernel 的延迟
"""
import torch
import time
import sys
import os

sys.path.insert(0, ".")
import GPT as gpt_module

# 配置小模型以减少 kernel 执行时间
gpt_module.emb_size = 128
gpt_module.head_size = 16
gpt_module.n_layer = 1
gpt_module.sequence_len = 64

from GPT import CharGPT

def measure_op_interval_native():
    """原生 PyTorch（无调度器）的算子间隔"""
    model = CharGPT(vs=100).cuda().eval()
    dummy = torch.randint(0, 100, (2, 64), device='cuda')

    # Warmup
    for _ in range(5):
        _ = model(dummy)
    torch.cuda.synchronize()

    # 测量：连续 100 次 forward
    num_iters = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(num_iters):
        _ = model(dummy)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time_ms = (t1 - t0) * 1000
    avg_time_us = total_time_ms * 1000 / num_iters

    print(f"Native PyTorch (no scheduler):")
    print(f"  Total: {total_time_ms:.2f} ms")
    print(f"  Avg per forward: {avg_time_us:.1f} μs")
    print(f"  Throughput: {num_iters / (t1 - t0):.0f} ops/sec")
    return avg_time_us

def measure_op_interval_with_scheduler():
    """带调度器的算子间隔（需要 LD_PRELOAD）"""
    try:
        import ctypes
        lib_path = "../build/libgpu_scheduler.so"
        if not os.path.exists(lib_path):
            print("Scheduler library not found, skipping")
            return None

        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        lib.orion_init_scheduler.argtypes = [ctypes.c_int]
        lib.orion_init_scheduler.restype = ctypes.c_int
        lib.orion_start_scheduler_thread.argtypes = []
        lib.orion_start_scheduler_thread.restype = ctypes.c_int
        lib.orion_set_client_idx.argtypes = [ctypes.c_int]
        lib.orion_set_client_idx.restype = None
        lib.orion_stop_scheduler.restype = None

        # 初始化调度器（1 个 client）
        ret = lib.orion_init_scheduler(1)
        if ret != 0:
            print("Failed to init scheduler")
            return None

        lib.orion_start_scheduler_thread()
        lib.orion_set_client_idx(0)

        model = CharGPT(vs=100).cuda().eval()
        dummy = torch.randint(0, 100, (2, 64), device='cuda')

        # Warmup
        for _ in range(5):
            _ = model(dummy)
        torch.cuda.synchronize()

        # 测量
        num_iters = 100
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(num_iters):
            _ = model(dummy)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        lib.orion_stop_scheduler()

        total_time_ms = (t1 - t0) * 1000
        avg_time_us = total_time_ms * 1000 / num_iters

        print(f"\nWith Orion Scheduler (multi-worker):")
        print(f"  Total: {total_time_ms:.2f} ms")
        print(f"  Avg per forward: {avg_time_us:.1f} μs")
        print(f"  Throughput: {num_iters / (t1 - t0):.0f} ops/sec")
        return avg_time_us

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    print("="*60)
    print("算子间隔测试（单 client 连续提交）")
    print("="*60)

    native_time = measure_op_interval_native()
    scheduler_time = measure_op_interval_with_scheduler()

    if scheduler_time:
        overhead = scheduler_time - native_time
        overhead_pct = (overhead / native_time) * 100
        print(f"\n调度器开销:")
        print(f"  绝对值: {overhead:.1f} μs")
        print(f"  相对值: {overhead_pct:.1f}%")

        if overhead < 5:
            print(f"  结论: 开销可忽略不计 (< 5μs)")
        elif overhead < 20:
            print(f"  结论: 开销较小 (5-20μs)")
        else:
            print(f"  结论: 开销显著 (> 20μs)")
