#!/bin/bash
# Orion GPU Scheduler 启动脚本
# 用法: ./run.sh [--num-be N] [--num-iters N] [--debug]

set -e

# 项目路径
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIB_PATH="$PROJECT_DIR/build/libgpu_scheduler.so"

# 环境变量 - 必须在最前面设置
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.1}
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$PATH

# Green Context 自动调优（根据 kernel_info 自动选择最优 SM 分配）
export ORION_GC_AUTOTUNE=1

# 检查库是否存在
if [ ! -f "$LIB_PATH" ]; then
    echo "Library not found at $LIB_PATH"
    echo "Building..."
    cd "$PROJECT_DIR"
    CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn make
fi

# 解析参数
DEBUG_MODE=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG_MODE="ORION_LOG_LEVEL=DEBUG"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# 日志文件配置
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/orion_${TIMESTAMP}.log"

# 运行测试
cd "$PROJECT_DIR/python"
echo "=============================================="
echo "Running Orion Scheduling Test (GPT + VGG)"
echo "=============================================="
echo "Library: $LIB_PATH"
echo "Arguments: $EXTRA_ARGS"
echo "Log file: $LOG_FILE"
echo "Green Context Auto-tune: $ORION_GC_AUTOTUNE"
echo "Profiler: PyTorch Profiler (Chrome trace)"
echo ""

if [ -n "$DEBUG_MODE" ]; then
    env $DEBUG_MODE LD_PRELOAD="$LIB_PATH" \
        python3 test_orion_blocking.py $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"
else
    LD_PRELOAD="$LIB_PATH" \
        python3 test_orion_blocking.py $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "Log saved to: $LOG_FILE"

