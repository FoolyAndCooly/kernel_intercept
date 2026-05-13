# 使用 NVIDIA PyTorch 官方镜像（已包含 CUDA 13.x + PyTorch + cuDNN）
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-13.1 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/lib:${LD_LIBRARY_PATH} \
    PATH=/usr/local/cuda-13.1/bin:${PATH}

# 安装项目特定依赖
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    matplotlib

WORKDIR /workspace/orion
CMD ["/bin/bash"]
