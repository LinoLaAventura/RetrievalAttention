FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Define environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    make \
    cmake \
    ninja-build \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils python${PYTHON_VERSION}-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Upgrade pip and install essential python build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# Set working directory
WORKDIR /workspace

# Install PyTorch for CUDA 12.4
# (Adjust version if necessary, typically torch==2.5.* based on README)
RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install flashinfer specifically for cu124/torch2.5 as requested in README
RUN pip install --no-cache-dir flashinfer-python==0.2.4 -i https://flashinfer.ai/whl/cu124/torch2.5/

# Copy the requirements file and install dependencies
# We assume you will mount or copy your code into /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and install the custom weighted flash attention (we do it from Git directly)
# or you can copy it if it's in the repo. Here we clone it for a clean build.
RUN git clone https://github.com/microsoft/flash-attention.git -b "@weighted" /workspace/flash-attention && \
    cd /workspace/flash-attention/csrc/cutlass && \
    git checkout 4e99f06 && \
    cd /workspace/flash-attention && \
    python setup.py install

# Copy project files
COPY . /workspace

# Build local library kernels
RUN cd /workspace/library/retroinfer && \
    python setup.py install

# Define default command
CMD ["bash"]
