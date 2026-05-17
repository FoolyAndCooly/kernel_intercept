#!/bin/bash
# 测试 Level 1 Worker 异步模式
# 用法: ./test_async.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_PATH="$PROJECT_DIR/build/libgpu_scheduler.so"

# 环境变量
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.1}
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH

# 检查库是否存在
if [ ! -f "$LIB_PATH" ]; then
    echo "Library not found at $LIB_PATH"
    echo "Building..."
    cd "$PROJECT_DIR"
    CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn make
fi

echo "=============================================="
echo "Level 1 Worker 异步模式测试"
echo "=============================================="
echo ""

# 测试 1: 同步模式 (baseline)
echo "测试 1: 同步模式 (ORION_ASYNC_MODE=0)"
echo "----------------------------------------------"
cd "$PROJECT_DIR/python"
ORION_ASYNC_MODE=0 LD_PRELOAD="$LIB_PATH" python3 test_async_mode.py

echo ""
echo "=============================================="
echo "测试完成"
echo "=============================================="
