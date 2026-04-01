# syntax=docker/dockerfile:1
# RetroInfer 项目专用 Dockerfile（稳健版）
# 目标：尽量提升一次性构建成功率，并把“会卡住/会失败”的点前置说明。

ARG BASE_IMAGE=hub.1panel.dev/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_MIRROR=
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG HF_ENDPOINT=https://hf-mirror.com

# 可选开关：
# 1) 如果源码里没有 `library/cutlass`，是否允许在构建时在线 clone
ARG ALLOW_CUTLASS_CLONE=1
# 2) 是否在构建期编译 retroinfer CUDA 扩展（建议保持 1）
ARG BUILD_RETROINFER_KERNELS=1
# 3) 是否安装 reasoning 相关依赖（只做简单推理可设为 0）
ARG INSTALL_REASONING_DEPS=1

ENV HF_ENDPOINT=${HF_ENDPOINT}
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=8
# 必须在构建扩展前设置，否则扩展可能不包含目标 GPU 架构
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"

WORKDIR /workspace/RetroInfer

# 1. 配置 apt 镜像（可选）并安装系统依赖
RUN set -eux; \
    if [[ -n "${APT_MIRROR}" ]]; then \
      sed -i "s|http://archive.ubuntu.com/ubuntu/|${APT_MIRROR}|g" /etc/apt/sources.list; \
      sed -i "s|http://security.ubuntu.com/ubuntu/|${APT_MIRROR}|g" /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget git \
      build-essential gcc-12 g++-12 \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      libstdc++6 libopenblas-dev libssl-dev; \
    rm -rf /var/lib/apt/lists/*

# 2. 设置 Python 版本并升级 pip
RUN set -eux; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1; \
    python3 -m pip install --upgrade pip==25.0

# 3. 先只复制依赖清单，利用 Docker layer cache
COPY requirements.txt /workspace/RetroInfer/requirements.txt

# 4. 安装 Python 依赖（带重试）
# 说明：torch / nvidia wheel 体积很大，网络不稳定时这是最容易卡住的阶段。
RUN set -eux; \
    retry() { n=0; until [[ $n -ge 5 ]]; do "$@" && break; n=$((n+1)); echo "retry $n/5: $*"; sleep 10; done; [[ $n -lt 5 ]]; }; \
    if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then \
      retry pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" -r /workspace/RetroInfer/requirements.txt; \
      retry pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" flash-attn==2.7.3 --no-build-isolation; \
      retry pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/; \
    else \
      retry pip install -r /workspace/RetroInfer/requirements.txt; \
      retry pip install flash-attn==2.7.3 --no-build-isolation; \
      retry pip install flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/; \
    fi

# 5. 复制完整源码
COPY . /workspace/RetroInfer

# 6. 确保 cutlass 可用（不再静默忽略失败）
# - 若仓库已包含 `library/cutlass`，直接使用。
# - 若未包含且 ALLOW_CUTLASS_CLONE=1，则尝试在线拉取（带重试）。
# - 若仍不可用，直接失败，避免后面编译阶段才报晦涩错误。
RUN set -eux; \
    if [[ -d library/cutlass ]]; then \
      echo "cutlass already present in source tree"; \
    else \
      if [[ "${ALLOW_CUTLASS_CLONE}" != "1" ]]; then \
        echo "ERROR: library/cutlass missing and ALLOW_CUTLASS_CLONE=0" >&2; \
        exit 1; \
      fi; \
      retry() { n=0; until [[ $n -ge 5 ]]; do "$@" && break; n=$((n+1)); echo "retry $n/5: $*"; sleep 10; done; [[ $n -lt 5 ]]; }; \
      retry git clone https://github.com/NVIDIA/cutlass.git library/cutlass; \
    fi

# 7. 编译 RetroInfer CUDA 扩展
# 这是容器可运行的关键步骤之一：不编译通常会在运行时报 ImportError/符号错误。
RUN set -eux; \
    if [[ "${BUILD_RETROINFER_KERNELS}" == "1" ]]; then \
      cd library/retroinfer; \
      CUDA_HOME=/usr/local/cuda CC=gcc-12 CXX=g++-12 pip install --no-build-isolation .; \
    else \
      echo "WARNING: BUILD_RETROINFER_KERNELS=0, runtime may fail if extension is required"; \
    fi

# 8. 可选安装 Reasoning Benchmark 依赖
RUN set -eux; \
    if [[ "${INSTALL_REASONING_DEPS}" == "1" ]]; then \
      cd benchmark/reasoning/latex2sympy && pip install -e . && cd ../../..; \
      pip install -r benchmark/reasoning/requirements.txt; \
    else \
      echo "skip reasoning dependencies"; \
    fi

# 9. 轻量验收（仅导入，不下载模型）
RUN python3 - <<'PY'
import importlib
importlib.import_module('weighted_flash_decoding')
importlib.import_module('model_hub')
print('image sanity check ok')
PY

WORKDIR /workspace/RetroInfer
CMD ["/bin/bash"]

# 关键说明：
# - 不能承诺“100% 不踩雷”，但本文件已把高频失败点（网络重试、关键依赖、静默失败）尽量前置和显式化。
# - 完整 `simple_test.py` 首次运行仍可能因 HuggingFace 模型下载耗时而看起来“卡住”，建议挂载本地 HF 缓存：
#   `-v /your/cache:/root/.cache/huggingface`
