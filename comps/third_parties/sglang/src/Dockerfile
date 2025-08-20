# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]

ARG VER_SGLANG=main
ARG VER_TORCH=2.7.1
ARG VER_TORCHVISION=0.22.1
ARG VER_TRITON=3.3.1

RUN apt-get update && \
    apt-get full-upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    curl \
    wget \
    vim \
    gcc \
    g++ \
    make

WORKDIR /sgl-workspace

RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm -f miniforge.sh && \
    . miniforge3/bin/activate && \
    conda install -y libsqlite==3.48.0 gperftools tbb libnuma numactl

ENV PATH=/sgl-workspace/miniforge3/bin:/sgl-workspace/miniforge3/condabin:${PATH}
ENV PIP_ROOT_USER_ACTION=ignore
ENV CONDA_PREFIX=/sgl-workspace/miniforge3

RUN pip config set global.index-url https://download.pytorch.org/whl/cpu && \
    pip config set global.extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir intel-openmp

RUN git clone https://github.com/sgl-project/sglang.git && \
    cd sglang && \
    git checkout ${VER_SGLANG} && \
    pip install --no-cache-dir -e "python[all_cpu]" && \
    pip install --no-cache-dir torch==${VER_TORCH} torchvision==${VER_TORCHVISION} triton==${VER_TRITON} --force-reinstall && \
    cd sgl-kernel && \
    cp pyproject_cpu.toml pyproject.toml && \
    pip install --no-cache-dir -v .

ENV SGLANG_USE_CPU_ENGINE=1
ENV LD_PRELOAD=/sgl-workspace/miniforge3/lib/libiomp5.so:/sgl-workspace/miniforge3/lib/libtcmalloc.so:/sgl-workspace/miniforge3/lib/libtbbmalloc.so.2

WORKDIR /sgl-workspace/sglang

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python3 -m sglang.launch_server --model ${MODEL_ID} --trust-remote-code --device cpu --disable-overlap-schedule --chunked-prefill-size 2048 --max-running-requests 32 --mem-fraction-static 0.8 --context-length 65536 --max-total-tokens 65536 --host 0.0.0.0 --port ${SGLANG_LLM_PORT} --api-key ${HF_TOKEN} --chat-template llama-4"]

