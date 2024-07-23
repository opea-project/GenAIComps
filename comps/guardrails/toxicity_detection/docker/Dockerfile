# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:22.04

ARG TAG=main

RUN apt-get update \ 
    && apt-get upgrade -y \ 
    && apt-get install -y --no-install-recommends \
    gcc-12 \
    g++-12 \
    make \
    wget \
    libnuma-dev \
    numactl \
    git \
    pkg-config \
    software-properties-common \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libsqlite3-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install python
WORKDIR /tmp
RUN wget -q https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz \
    && tar -xzvf Python-3.8.10.tgz
WORKDIR /tmp/Python-3.8.10
RUN ./configure --prefix=/usr/bin/python3.8 --enable-optimizations \
    && make -j \
    && make install \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.8/bin/python3.8 60 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/python3.8/bin/pip3 60 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir wheel \
    && rm -rf /tmp/* \
    && echo "export PATH=/usr/bin/python3.8:\$PATH" >> ~/.bashrc

RUN pip install --no-cache-dir torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir cmake==3.26.1 transformers==4.41.2 sentencepiece==0.1.99 accelerate==0.23.0 protobuf tiktoken transformers-stream-generator einops \
    && ln -s /usr/bin/python3.8/lib/python3.8/site-packages/cmake/data/bin/cmake /usr/bin/cmake

COPY comps /root/comps

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /root/comps/guardrails/toxicity_detection/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/root

RUN chmod +x /root/comps/guardrails/toxicity_detection/run.sh

WORKDIR /root/comps/guardrails/toxicity_detection/

ENTRYPOINT ["/root/comps/guardrails/toxicity_detection/run.sh"]