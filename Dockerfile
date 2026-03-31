# RetroInfer 项目专用 Dockerfile
# =====================================
# 该 Dockerfile 旨在最大程度还原论文作者的开发环境，
# 并自动解决原项目中未声明的依赖和国内网络问题。
# =====================================

# 1. 选择官方推荐的 CUDA 基础镜像（含 cuDNN）
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 2. 安装系统级依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget curl build-essential gcc-12 g++-12 python3.10 python3.10-venv python3.10-dev \
        ca-certificates libstdc++6 libopenblas-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# 3. 设置 Python3.10 为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python3 -m pip install --upgrade pip==25.0

# 4. 创建工作目录
WORKDIR /workspace/RetroInfer

# 5. 复制项目代码到容器
COPY . /workspace/RetroInfer

# 6. 配置国内 HuggingFace 镜像加速（如需可注释）
ENV HF_ENDPOINT=https://hf-mirror.com

# 7. 安装 Python 依赖
RUN pip install -r requirements.txt \
    && pip install flash-attn==2.7.3 --no-build-isolation \
    && pip install flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/

# 8. 拉取并编译定制 flash-attention（weighted 分支）
RUN git clone -b weighted https://github.com/Starmys/flash-attention.git /tmp/flash-attn && \
    cd /tmp/flash-attn && pip install --no-build-isolation . && cd /workspace/RetroInfer && rm -rf /tmp/flash-attn

# 9. 拉取并补全缺失的 cutlass 依赖（RetroInfer kernels 必需）
RUN git clone https://github.com/NVIDIA/cutlass.git library/cutlass

# 10. 编译 RetroInfer 自定义 CUDA 内核
RUN cd library/retroinfer && \
    CUDA_HOME=/usr/local/cuda \
    CC=gcc-12 CXX=g++-12 \
    pip install --no-build-isolation .

# 11. 安装 Reasoning Benchmark 依赖
RUN cd benchmark/reasoning/latex2sympy && pip install -e . && cd ../../..
RUN pip install -r benchmark/reasoning/requirements.txt

# 12. 设置默认工作目录
WORKDIR /workspace/RetroInfer

# 13. 容器启动后默认进入 bash
CMD ["/bin/bash"]

# ========== 说明 ==========
# - 该 Dockerfile 兼容论文所有评测脚本（RULER/LongBench/Reasoning）。
# - 已自动解决 cutlass、flash-attention@weighted、国内镜像等常见环境坑。
# - 如需挂载本地模型缓存，运行时加 -v /your/cache:/root/.cache/huggingface。
# - 如需自定义 CUDA 版本或 Python 版本，请相应调整 FROM 和 apt-get 部分。
# ==========================
