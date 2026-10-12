FROM ghcr.io/onnxmlir/onnx-mlir-dev
WORKDIR /workdir
ENV HOME=/workdir

ENV PATH=$PATH:/workdir/bin

# 1) Core Utilities & Debugging Tools
RUN apt-get update && apt-get install -y \
    python3-numpy \
    python3-pip \
    gdb \
    lldb \
    valgrind \
    vim \
    clang-format \
    libeigen3-dev \
    git \
    ssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python3 -m pip install --no-deps --ignore-installed wheel --break-system-packages

# 2) ONNX-MLIR Working Directory
WORKDIR /workdir/onnx-mlir

# 3) Environment Settings for Debug Mode
WORKDIR /workdir
ENV NPROC=4
ENV PATH=$PATH:/workdir/onnx-mlir/build/Debug/bin/:/workdir/onnx-mlir/build/Debug/lib:/workdir/llvm-project/build/bin
