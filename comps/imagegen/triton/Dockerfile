# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG TRITON_VERSION=24.04

FROM nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3 AS triton

FROM base

ARG MODEL_NAME=stability

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt update && \
    apt install -y datacenter-gpu-manager

RUN apt update && \
    apt install -y libb64-0d \
              build-essential \
              libaio-dev \
              libaio1 \
              libsndfile-dev \
              libcupti-dev \
              libjpeg-dev \
              libpng-dev \
              libwebp-dev 

ARG TRITON_VERSION=24.04

COPY --from=triton /opt/tritonserver /opt/tritonserver
COPY --from=triton /usr/local/cuda-* /usr/local/cuda

ENV PATH=$PATH:/opt/tritonserver/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tritonserver/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib

RUN git clone https://github.com/triton-inference-server/python_backend -b r${TRITON_VERSION} /opt/tritonserver/backends/opea_backends && \
    mkdir -p /opt/tritonserver/backends/opea_backends/models/${MODEL_NAME}/1

COPY ./model.py /opt/tritonserver/backends/opea_backends/models/${MODEL_NAME}/1/model.py
COPY ./config.pbtxt /opt/tritonserver/backends/opea_backends/models/${MODEL_NAME}/config.pbtxt

CMD ["tritonserver", "--model-repository", "/opt/tritonserver/backends/opea_backends/models"]
