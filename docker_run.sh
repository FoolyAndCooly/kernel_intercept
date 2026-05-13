#!/bin/bash
# 构建并启动 Orion GPU Scheduler Docker 容器
# 用法: bash docker_run.sh

set -e

IMAGE_NAME="orion-scheduler"
CONTAINER_NAME="orion-dev"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Building Docker image ==="
docker build -t "$IMAGE_NAME" "$PROJECT_DIR"
echo "✓ Image built: $IMAGE_NAME"
echo ""

# 检查容器是否已存在
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "=== Container '$CONTAINER_NAME' already exists ==="
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "✓ Container is running"
    else
        echo "Starting existing container..."
        docker start "$CONTAINER_NAME"
        echo "✓ Container started"
    fi
else
    echo "=== Step 2: Creating and starting container ==="
    docker run -d \
        --gpus all \
        --name "$CONTAINER_NAME" \
        -v /usr/local/cuda-13.1:/usr/local/cuda-13.1 \
        -v /usr/local/cuda/bin/ncu:/usr/local/cuda/bin/ncu:ro \
        -e CUDA_HOME=/usr/local/cuda-13.1 \
        -e LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/lib \
        -e PATH=/usr/local/cuda-13.1/bin:/usr/local/bin:/usr/bin:/bin \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        -v "$PROJECT_DIR":/workspace/orion \
        -w /workspace/orion \
        "$IMAGE_NAME" \
        tail -f /dev/null
    echo "✓ Container created and started: $CONTAINER_NAME"

    # 创建 CUDA 12 兼容性符号链接
    echo "Creating CUDA 12 compatibility symlinks..."
    docker exec "$CONTAINER_NAME" bash -c "cd /usr/local/cuda-13.1/lib64 && ln -sf libcudart.so.13 libcudart.so.12 && ln -sf libcudart.so.13.1.80 libcudart.so.12.0.0"
    echo "✓ Symlinks created"
fi

echo ""
echo "=== Setup Complete ==="
echo "Container: $CONTAINER_NAME"
echo "Project:   /workspace/orion"
echo "CUDA:      13.1 (from host /usr/local/cuda-13.1)"
echo ""
echo "To enter the container:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To remove the container:"
echo "  docker rm -f $CONTAINER_NAME"
