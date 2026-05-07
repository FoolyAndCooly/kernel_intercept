#!/usr/bin/env python3
"""
基础功能测试脚本（重构后）

测试内容：
1. 调度器启动/停止
2. 单客户端 kernel 执行
3. 多客户端并发执行

使用方法：
    LD_PRELOAD=./build/libgpu_scheduler.so python3 test/test_basic.py
"""

import torch
import ctypes
import os
import sys
import threading
import time

# 设置日志级别
os.environ['ORION_LOG_LEVEL'] = '3'  # INFO level

def load_library():
    """加载调度器库"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    lib_path = os.path.join(project_root, "build", "libgpu_scheduler.so")

    if not os.path.exists(lib_path):
        print(f"ERROR: Library not found at {lib_path}")
        sys.exit(1)

    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

    # 基本接口
    lib.orion_start_scheduler.argtypes = [ctypes.c_int]
    lib.orion_start_scheduler.restype = ctypes.c_int
    lib.orion_stop_scheduler.restype = None
    lib.orion_set_client_idx.argtypes = [ctypes.c_int]
    lib.orion_set_client_idx.restype = None

    # Orion 调度接口
    lib.orion_load_kernel_info.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.orion_load_kernel_info.restype = ctypes.c_int
    lib.orion_set_client_kernels.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.orion_set_client_kernels.restype = None
    lib.orion_set_sm_threshold.argtypes = [ctypes.c_int]
    lib.orion_set_sm_threshold.restype = None
    lib.orion_get_sm_threshold.argtypes = []
    lib.orion_get_sm_threshold.restype = ctypes.c_int
    lib.orion_reset_state.restype = None

    print("[OK] Library loaded successfully")
    return lib


def test_scheduler_start_stop(lib):
    """测试1: 调度器启动和停止"""
    print("\n" + "="*60)
    print("Test 1: Scheduler Start/Stop")
    print("="*60)

    # 启动调度器
    ret = lib.orion_start_scheduler(2)
    if ret != 0:
        print("[FAIL] Failed to start scheduler")
        return False
    print("[OK] Scheduler started with 2 clients")

    # 获取 SM 阈值
    sm_threshold = lib.orion_get_sm_threshold()
    print(f"[OK] SM threshold: {sm_threshold}")

    # 停止调度器
    lib.orion_stop_scheduler()
    print("[OK] Scheduler stopped")

    return True


def test_single_client(lib):
    """测试2: 单客户端执行"""
    print("\n" + "="*60)
    print("Test 2: Single Client Execution")
    print("="*60)

    # 启动调度器
    ret = lib.orion_start_scheduler(1)
    if ret != 0:
        print("[FAIL] Failed to start scheduler")
        return False

    # 设置客户端索引
    lib.orion_set_client_idx(0)
    print("[OK] Client index set to 0 (HP)")

    # 创建张量并执行操作
    print("[INFO] Running CUDA operations...")
    try:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')

        start = time.time()
        for i in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000

        print(f"[OK] 10x matmul completed in {elapsed:.2f} ms")
    except Exception as e:
        print(f"[FAIL] CUDA operation failed: {e}")
        lib.orion_stop_scheduler()
        return False

    lib.orion_stop_scheduler()
    print("[OK] Test passed")
    return True


def test_multi_client(lib):
    """测试3: 多客户端并发执行"""
    print("\n" + "="*60)
    print("Test 3: Multi-Client Concurrent Execution")
    print("="*60)

    NUM_CLIENTS = 2
    NUM_ITERS = 5

    # 启动调度器
    ret = lib.orion_start_scheduler(NUM_CLIENTS)
    if ret != 0:
        print("[FAIL] Failed to start scheduler")
        return False
    print(f"[OK] Scheduler started with {NUM_CLIENTS} clients")

    # 设置 SM 阈值
    lib.orion_set_sm_threshold(50)
    print(f"[OK] SM threshold set to 50")

    # 准备数据
    inputs = []
    for i in range(NUM_CLIENTS):
        inputs.append({
            'a': torch.randn(500, 500, device='cuda'),
            'b': torch.randn(500, 500, device='cuda')
        })
    torch.cuda.synchronize()

    # 同步原语
    barrier = threading.Barrier(NUM_CLIENTS)
    results = [None] * NUM_CLIENTS
    errors = [None] * NUM_CLIENTS

    def client_worker(client_idx):
        try:
            lib.orion_set_client_idx(client_idx)
            client_type = "HP" if client_idx == 0 else f"BE{client_idx}"

            # 等待所有线程就绪
            barrier.wait()

            start = time.time()
            for i in range(NUM_ITERS):
                c = torch.matmul(inputs[client_idx]['a'], inputs[client_idx]['b'])
            torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000

            results[client_idx] = elapsed
            print(f"[OK] {client_type}: {NUM_ITERS}x matmul in {elapsed:.2f} ms")
        except Exception as e:
            errors[client_idx] = str(e)
            print(f"[FAIL] Client {client_idx} error: {e}")

    # 启动线程
    threads = []
    for i in range(NUM_CLIENTS):
        t = threading.Thread(target=client_worker, args=(i,))
        threads.append(t)
        t.start()

    # 等待完成
    for t in threads:
        t.join()

    lib.orion_stop_scheduler()

    # 检查结果
    if any(e is not None for e in errors):
        print("[FAIL] Some clients failed")
        return False

    print("[OK] All clients completed successfully")
    return True


def test_memory_operations(lib):
    """测试4: 内存操作"""
    print("\n" + "="*60)
    print("Test 4: Memory Operations")
    print("="*60)

    ret = lib.orion_start_scheduler(1)
    if ret != 0:
        print("[FAIL] Failed to start scheduler")
        return False

    lib.orion_set_client_idx(0)

    try:
        # cudaMalloc
        print("[INFO] Testing cudaMalloc...")
        a = torch.empty(1000, 1000, device='cuda')
        print("[OK] cudaMalloc succeeded")

        # cudaMemcpy (H2D)
        print("[INFO] Testing cudaMemcpy H2D...")
        b = torch.randn(1000, 1000)
        a.copy_(b)
        print("[OK] cudaMemcpy H2D succeeded")

        # cudaMemcpy (D2H)
        print("[INFO] Testing cudaMemcpy D2H...")
        c = a.cpu()
        print("[OK] cudaMemcpy D2H succeeded")

        # cudaMemset
        print("[INFO] Testing cudaMemset...")
        a.zero_()
        print("[OK] cudaMemset succeeded")

        # cudaFree
        print("[INFO] Testing cudaFree...")
        del a
        torch.cuda.empty_cache()
        print("[OK] cudaFree succeeded")

    except Exception as e:
        print(f"[FAIL] Memory operation failed: {e}")
        lib.orion_stop_scheduler()
        return False

    lib.orion_stop_scheduler()
    print("[OK] Test passed")
    return True


def main():
    print("="*60)
    print("Orion Scheduler Basic Tests (Post-Refactor)")
    print("="*60)

    lib = load_library()

    tests = [
        ("Scheduler Start/Stop", test_scheduler_start_stop),
        ("Single Client", test_single_client),
        ("Multi-Client", test_multi_client),
        ("Memory Operations", test_memory_operations),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func(lib)
            results.append((name, passed))
        except Exception as e:
            print(f"[FAIL] Test '{name}' crashed: {e}")
            results.append((name, False))
        time.sleep(0.5)  # 等待资源释放

    # 打印总结
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {status} {name}")

    print("-"*60)
    print(f"  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
