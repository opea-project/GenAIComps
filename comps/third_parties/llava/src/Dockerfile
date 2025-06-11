# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set this to "cpu" or "gpu" or etc
ARG ARCH="cpu"

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing curl

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user:/usr/lib/habanalabs/:/optimum-habana
ARG ARCH=cpu

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      uv pip install --no-cache-dir -r /home/user/comps/third_parties/llava/src/requirements-cpu.txt; \
    else \
      uv pip install --no-cache-dir -r /home/user/comps/third_parties/llava/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user
WORKDIR /home/user/comps/third_parties/llava/src

ENTRYPOINT ["python", "llava_server.py", "--device", "cpu"]
