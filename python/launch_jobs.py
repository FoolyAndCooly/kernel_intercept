#!/usr/bin/env python3
"""
Orion-Style GPU Scheduler - PyTorch Integration Launcher

Usage:
    python launch_jobs.py --hp-model resnet50 --be-model vgg16 --num-iters 100

Environment Variables:
    ORION_LIB_PATH: Path to libgpu_scheduler.so (default: ../build/libgpu_scheduler.so)
    ORION_LOG_LEVEL: Log level (0=NONE, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=TRACE)
    ORION_PROFILE_PATH: Path to kernel profile JSON file
"""

import os
import sys
import time
import ctypes
import argparse
import threading
import multiprocessing as mp
from typing import Optional, List, Callable

# 确保可以导入 torch
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("Error: PyTorch not found. Please install PyTorch first.")
    sys.exit(1)


class OrionScheduler:
    """Orion 调度器的 Python 封装"""
    
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or os.environ.get(
            "ORION_LIB_PATH", 
            os.path.join(os.path.dirname(__file__), "..", "build", "libgpu_scheduler.so")
        )
        self.lib = None
        self.initialized = False
        
    def load_library(self) -> bool:
        """加载调度器动态库"""
        if not os.path.exists(self.lib_path):
            print(f"Error: Library not found at {self.lib_path}")
            return False
            
        try:
            # 使用 RTLD_GLOBAL 确保符号可被其他库访问
            self.lib = ctypes.CDLL(self.lib_path, mode=ctypes.RTLD_GLOBAL)
            
            # 设置函数签名
            self.lib.orion_init.argtypes = [ctypes.c_int]
            self.lib.orion_init.restype = ctypes.c_int
            
            self.lib.orion_shutdown.argtypes = []
            self.lib.orion_shutdown.restype = None
            
            self.lib.orion_set_client_idx.argtypes = [ctypes.c_int]
            self.lib.orion_set_client_idx.restype = None
            
            self.lib.orion_get_client_idx.argtypes = []
            self.lib.orion_get_client_idx.restype = ctypes.c_int
            
            self.lib.orion_set_enabled.argtypes = [ctypes.c_int]
            self.lib.orion_set_enabled.restype = None
            
            self.lib.orion_start_scheduler.argtypes = [ctypes.c_int]
            self.lib.orion_start_scheduler.restype = ctypes.c_int
            
            self.lib.orion_stop_scheduler.argtypes = []
            self.lib.orion_stop_scheduler.restype = None
            
            self.lib.block.argtypes = [ctypes.c_int]
            self.lib.block.restype = None
            
            print(f"Loaded Orion library from {self.lib_path}")
            return True
            
        except Exception as e:
            print(f"Error loading library: {e}")
            return False
    
    def init(self, num_clients: int) -> bool:
        """初始化调度器"""
        if not self.lib:
            if not self.load_library():
                return False
                
        ret = self.lib.orion_start_scheduler(num_clients)
        if ret == 0:
            self.initialized = True
            print(f"Initialized Orion scheduler with {num_clients} clients")
            return True
        else:
            print("Failed to initialize Orion scheduler")
            return False
    
    def shutdown(self):
        """关闭调度器"""
        if self.lib and self.initialized:
            self.lib.orion_stop_scheduler()
            self.initialized = False
            print("Orion scheduler shutdown")
    
    def set_client_idx(self, idx: int):
        """设置当前线程的 client index"""
        if self.lib:
            self.lib.orion_set_client_idx(idx)
    
    def get_client_idx(self) -> int:
        """获取当前线程的 client index"""
        if self.lib:
            return self.lib.orion_get_client_idx()
        return -1
    
    def set_enabled(self, enabled: bool):
        """启用/禁用拦截"""
        if self.lib:
            self.lib.orion_set_enabled(1 if enabled else 0)
    
    def block(self, phase: int = 0):
        """阻塞等待调度器"""
        if self.lib:
            self.lib.block(phase)


# 全局调度器实例
scheduler = OrionScheduler()


def client_worker(
    client_idx: int,
    model_fn: Callable,
    num_iters: int,
    batch_size: int,
    device: str,
    result_queue: mp.Queue,
    start_barrier: threading.Barrier,
    end_barrier: threading.Barrier
):
    """
    Client 工作线程
    
    Args:
        client_idx: Client 索引 (0 = HP, 1+ = BE)
        model_fn: 返回模型的函数
        num_iters: 迭代次数
        batch_size: 批量大小
        device: 设备 (cuda:0)
        result_queue: 结果队列
        start_barrier: 启动屏障
        end_barrier: 结束屏障
    """
    try:
        # 设置 client index
        scheduler.set_client_idx(client_idx)
        
        # 创建模型
        model = model_fn().to(device)
        model.eval()
        
        # 创建虚拟输入
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        
        # 等待所有 client 就绪
        start_barrier.wait()
        
        # 计时
        start_time = time.perf_counter()
        
        # 主循环
        with torch.no_grad():
            for i in range(num_iters):
                output = model(dummy_input)
                # 在每次迭代后同步
                scheduler.block(i)
                torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # 等待所有 client 完成
        end_barrier.wait()
        
        # 计算统计
        elapsed = end_time - start_time
        throughput = num_iters / elapsed
        latency = elapsed / num_iters * 1000  # ms
        
        result = {
            'client_idx': client_idx,
            'client_type': 'HP' if client_idx == 0 else 'BE',
            'num_iters': num_iters,
            'elapsed_time': elapsed,
            'throughput': throughput,
            'avg_latency_ms': latency
        }
        result_queue.put(result)
        
    except Exception as e:
        result_queue.put({'client_idx': client_idx, 'error': str(e)})


def create_resnet50():
    """创建 ResNet50 模型"""
    try:
        from torchvision.models import resnet50
        return resnet50(pretrained=False)
    except ImportError:
        # 如果没有 torchvision，使用简单的替代模型
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1000)
        )


def create_vgg16():
    """创建 VGG16 模型"""
    try:
        from torchvision.models import vgg16
        return vgg16(pretrained=False)
    except ImportError:
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1000)
        )


def create_simple_model():
    """创建简单的测试模型"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )


MODEL_REGISTRY = {
    'resnet50': create_resnet50,
    'vgg16': create_vgg16,
    'simple': create_simple_model,
}


def run_with_scheduler(
    hp_model_name: str,
    be_model_names: List[str],
    num_iters: int = 100,
    batch_size: int = 1,
    device: str = 'cuda:0'
):
    """
    使用 Orion 调度器运行多个模型
    
    Args:
        hp_model_name: 高优先级模型名称
        be_model_names: Best-effort 模型名称列表
        num_iters: 每个模型的迭代次数
        batch_size: 批量大小
        device: 设备
    """
    num_clients = 1 + len(be_model_names)
    
    print(f"\n{'='*60}")
    print("Orion GPU Scheduler - Multi-Model Benchmark")
    print(f"{'='*60}")
    print(f"HP Model: {hp_model_name}")
    print(f"BE Models: {be_model_names}")
    print(f"Iterations: {num_iters}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # 初始化调度器
    if not scheduler.init(num_clients):
        print("Failed to initialize scheduler")
        return
    
    try:
        # 准备模型函数
        model_fns = [MODEL_REGISTRY.get(hp_model_name, create_simple_model)]
        for name in be_model_names:
            model_fns.append(MODEL_REGISTRY.get(name, create_simple_model))
        
        # 创建同步原语
        result_queue = mp.Queue()
        start_barrier = threading.Barrier(num_clients)
        end_barrier = threading.Barrier(num_clients)
        
        # 创建工作线程
        threads = []
        for i, model_fn in enumerate(model_fns):
            t = threading.Thread(
                target=client_worker,
                args=(i, model_fn, num_iters, batch_size, device, 
                      result_queue, start_barrier, end_barrier)
            )
            threads.append(t)
        
        # 启动所有线程
        print("Starting worker threads...")
        for t in threads:
            t.start()
        
        # 等待完成
        for t in threads:
            t.join()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 打印结果
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        
        for r in sorted(results, key=lambda x: x.get('client_idx', -1)):
            if 'error' in r:
                print(f"Client {r['client_idx']}: ERROR - {r['error']}")
            else:
                print(f"Client {r['client_idx']} ({r['client_type']}):")
                print(f"  Throughput: {r['throughput']:.2f} iter/s")
                print(f"  Avg Latency: {r['avg_latency_ms']:.2f} ms")
                print(f"  Total Time: {r['elapsed_time']:.2f} s")
        
        print(f"{'='*60}\n")
        
    finally:
        scheduler.shutdown()


def run_baseline(
    model_name: str,
    num_iters: int = 100,
    batch_size: int = 1,
    device: str = 'cuda:0'
):
    """
    运行基线测试 (不使用调度器)
    """
    print(f"\n{'='*60}")
    print("Baseline Benchmark (No Scheduler)")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Iterations: {num_iters}")
    print(f"{'='*60}\n")
    
    model_fn = MODEL_REGISTRY.get(model_name, create_simple_model)
    model = model_fn().to(device)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # 计时
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    throughput = num_iters / elapsed
    latency = elapsed / num_iters * 1000
    
    print(f"Results:")
    print(f"  Throughput: {throughput:.2f} iter/s")
    print(f"  Avg Latency: {latency:.2f} ms")
    print(f"  Total Time: {elapsed:.2f} s")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Orion GPU Scheduler Launcher')
    
    parser.add_argument('--hp-model', type=str, default='simple',
                        help='High-priority model name')
    parser.add_argument('--be-models', type=str, nargs='+', default=['simple'],
                        help='Best-effort model names')
    parser.add_argument('--num-iters', type=int, default=100,
                        help='Number of iterations per model')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--baseline', action='store_true',
                        help='Run baseline without scheduler')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name in MODEL_REGISTRY:
            print(f"  - {name}")
        return
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        sys.exit(1)
    
    if args.baseline:
        run_baseline(args.hp_model, args.num_iters, args.batch_size, args.device)
    else:
        run_with_scheduler(
            args.hp_model,
            args.be_models,
            args.num_iters,
            args.batch_size,
            args.device
        )


if __name__ == '__main__':
    main()
