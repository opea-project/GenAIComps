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

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/third_parties/llava/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/third_parties/llava/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user

# air gapped support: set model cache dir
RUN mkdir -p /data && chown -R user /data
ENV HF_HUB_CACHE=/data
# air gapped support: download warmup jpg files
RUN mkdir -p /home/user/comps/third_parties/llava/src/cache && cd /home/user/comps/third_parties/llava/src/cache && curl -sJLO https://llava-vl.github.io/static/images/view.jpg
RUN chown -R user /home/user/comps/third_parties/llava/src/cache

USER user
WORKDIR /home/user/comps/third_parties/llava/src

ENTRYPOINT ["python", "llava_server.py", "--device", "cpu"]
