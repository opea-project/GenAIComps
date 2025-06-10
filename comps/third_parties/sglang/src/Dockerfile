# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS base

RUN apt-get update && \
    apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    curl \
    g++-11 \
    gcc-11 \
    git \
    make \
    numactl \
    wget

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100

WORKDIR /root

RUN curl -fsSL -v -o miniforge.sh -O https://github.com/conda-forge/miniforge/releases/download/24.7.1-2/Miniforge3-24.7.1-2-Linux-x86_64.sh && \
    bash miniforge.sh -b -p ./miniforge3 && \
    rm miniforge.sh

RUN git clone https://github.com/jianan-gu/sglang --depth=1 -b llama4_optimzed_cpu_r1 && \
    rm -rf sglang/.git
RUN . ~/miniforge3/bin/activate && conda create -n sglang python=3.10 && conda activate sglang && \
    pip install --no-cache-dir -e "python[all_cpu]" && \
    conda install -y libsqlite=3.48.0 && \
    pip uninstall -y triton && pip uninstall -y transformers && pip install --no-cache-dir transformers>=4.50.0 && \
    pip install --no-cache-dir triton==3.1 && pip install --no-cache-dir intel-openmp==2024.2.0 && pip install --no-cache-dir transformers

RUN git clone https://github.com/vllm-project/vllm.git --depth=1 -b v0.6.4.post1 && \
    rm -rf vllm/.git
RUN apt-get install -y libnuma-dev && \
    . ~/miniforge3/bin/activate && conda activate sglang && \
    pip install --no-cache-dir cmake==3.31.2 wheel packaging ninja "setuptools-scm>=8" numpy nvidia-ml-py && \
    pip install --no-cache-dir -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    VLLM_TARGET_DEVICE=cpu python setup.py develop

WORKDIR /root/sglang/sgl-kernel
RUN . ~/miniforge3/bin/activate && conda activate sglang && python setup.py install

WORKDIR /root/sglang
RUN . ~/miniforge3/bin/activate && conda activate sglang && pip uninstall -y torch torchvision && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu && \
    conda install -y gperftools gcc=11 gxx=11 cxx-compiler -c conda-forge

COPY ./comps/third_parties/sglang/src/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
