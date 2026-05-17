#!/usr/bin/env python3
"""
测试 Level 1 Worker 异步模式

验证：
1. 算子间隔显著降低（从 2-10μs 降到 ~150ns）
2. 错误能正确延迟到 cudaDeviceSynchronize
3. 结果正确性（与同步模式一致）
"""
import torch
import time
import sys
import os

sys.path.insert(0, ".")
import GPT as gpt_module

# 配置小模型
gpt_module.emb_size = 256
gpt_module.head_size = 32
gpt_module.n_layer = 2
gpt_module.sequence_len = 64

from GPT import CharGPT

def measure_throughput(async_mode, num_iters=100):
    """测量给定异步模式下的吞吐量"""
    try:
        import ctypes
        lib_path = "../build/libgpu_scheduler.so"
        if not os.path.exists(lib_path):
            print(f"Library not found: {lib_path}")
            return None

        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

        # 设置函数签名
        lib.orion_init_scheduler.argtypes = [ctypes.c_int]
        lib.orion_init_scheduler.restype = ctypes.c_int
        lib.orion_start_scheduler_thread.argtypes = []
        lib.orion_start_scheduler_thread.restype = ctypes.c_int
        lib.orion_set_client_idx.argtypes = [ctypes.c_int]
        lib.orion_set_client_idx.restype = None
        lib.orion_stop_scheduler.restype = None
        lib.orion_set_async_mode.argtypes = [ctypes.c_int]
        lib.orion_set_async_mode.restype = None
        lib.orion_get_async_mode.argtypes = []
        lib.orion_get_async_mode.restype = ctypes.c_int

        # 初始化调度器
        ret = lib.orion_init_scheduler(1)
        if ret != 0:
            print("Failed to init scheduler")
            return None

        # 设置异步模式
        lib.orion_set_async_mode(async_mode)
        actual_mode = lib.orion_get_async_mode()
        print(f"  Async mode set to: {actual_mode}")

        lib.orion_start_scheduler_thread()
        lib.orion_set_client_idx(0)

        # 创建模型
        model = CharGPT(vs=100).cuda().eval()
        dummy = torch.randint(0, 100, (2, 64), device='cuda')

        # Warmup
        for _ in range(5):
            _ = model(dummy)
        torch.cuda.synchronize()

        # 测量
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(num_iters):
            _ = model(dummy)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        lib.orion_stop_scheduler()

        total_time_ms = (t1 - t0) * 1000
        avg_time_us = total_time_ms * 1000 / num_iters
        throughput = num_iters / (t1 - t0)

        return {
            'total_ms': total_time_ms,
            'avg_us': avg_time_us,
            'throughput': throughput
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_error_propagation():
    """测试异步模式下的错误传播"""
    print("\n" + "="*60)
    print("测试错误传播（异步模式）")
    print("="*60)

    # TODO: 需要构造一个会失败的 kernel launch
    # 这里只是框架，实际测试需要更复杂的设置
    print("  [跳过] 需要构造失败场景")

def test_correctness():
    """测试异步模式结果正确性"""
    print("\n" + "="*60)
    print("测试结果正确性")
    print("="*60)

    try:
        import ctypes
        lib_path = "../build/libgpu_scheduler.so"
        if not os.path.exists(lib_path):
            print(f"Library not found: {lib_path}")
            return

        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        lib.orion_init_scheduler.argtypes = [ctypes.c_int]
        lib.orion_start_scheduler_thread.argtypes = []
        lib.orion_set_client_idx.argtypes = [ctypes.c_int]
        lib.orion_stop_scheduler.restype = None
        lib.orion_set_async_mode.argtypes = [ctypes.c_int]

        model = CharGPT(vs=100).cuda().eval()
        dummy = torch.randint(0, 100, (2, 64), device='cuda')

        # 同步模式结果
        lib.orion_init_scheduler(1)
        lib.orion_set_async_mode(0)
        lib.orion_start_scheduler_thread()
        lib.orion_set_client_idx(0)

        with torch.no_grad():
            result_sync = model(dummy)
        torch.cuda.synchronize()
        lib.orion_stop_scheduler()

        # 异步模式结果
        lib.orion_init_scheduler(1)
        lib.orion_set_async_mode(1)
        lib.orion_start_scheduler_thread()
        lib.orion_set_client_idx(0)

        with torch.no_grad():
            result_async = model(dummy)
        torch.cuda.synchronize()
        lib.orion_stop_scheduler()

        # 比较结果
        diff = torch.abs(result_sync - result_async).max().item()
        print(f"  Max difference: {diff}")

        if diff < 1e-5:
            print("  ✓ 结果一致")
        else:
            print(f"  ✗ 结果不一致 (diff={diff})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("="*60)
    print("Level 1 Worker 异步模式测试")
    print("="*60)

    # 测试吞吐量
    print("\n测试 1: 吞吐量对比")
    print("-"*60)

    print("\nLevel 0 (同步模式):")
    result_sync = measure_throughput(async_mode=0, num_iters=100)
    if result_sync:
        print(f"  总时间: {result_sync['total_ms']:.2f} ms")
        print(f"  平均延迟: {result_sync['avg_us']:.1f} μs")
        print(f"  吞吐量: {result_sync['throughput']:.0f} ops/sec")

    print("\nLevel 1 (Worker 异步):")
    result_async = measure_throughput(async_mode=1, num_iters=100)
    if result_async:
        print(f"  总时间: {result_async['total_ms']:.2f} ms")
        print(f"  平均延迟: {result_async['avg_us']:.1f} μs")
        print(f"  吞吐量: {result_async['throughput']:.0f} ops/sec")

    # 计算提升
    if result_sync and result_async:
        speedup = result_async['throughput'] / result_sync['throughput']
        latency_reduction = result_sync['avg_us'] - result_async['avg_us']
        print(f"\n性能提升:")
        print(f"  吞吐量提升: {speedup:.1f}x")
        print(f"  延迟降低: {latency_reduction:.1f} μs")

        if speedup > 5:
            print(f"  ✓ 显著提升 (>{speedup:.0f}x)")
        elif speedup > 2:
            print(f"  ✓ 明显提升 ({speedup:.1f}x)")
        else:
            print(f"  ⚠ 提升有限 ({speedup:.1f}x)")

    # 测试正确性
    test_correctness()

    # 测试错误传播
    test_error_propagation()

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
