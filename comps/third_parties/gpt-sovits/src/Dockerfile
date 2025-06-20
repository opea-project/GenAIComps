# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim

# Set this to "cpu" or "gpu" or etc
ARG ARCH="cpu"

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    curl \
    ffmpeg \
    git \
    git-lfs \
    google-perftools \
    libomp-dev \
    numactl \
    wget

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libiomp5.so:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
ENV MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LANG=C.UTF-8

# Clone source repo
RUN git clone --depth=1 --branch openai_compat --single-branch https://github.com/Spycsh/GPT-SoVITS.git

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r GPT-SoVITS/requirements.txt; \
    else \
      $uvpip -r GPT-SoVITS/requirements.txt; \
    fi

RUN python -m nltk.downloader averaged_perceptron_tagger_eng cmudict && \
    mv /root/nltk_data /home/user/ && \
    mv GPT-SoVITS /home/user/

COPY --chown=user:user comps/third_parties/gpt-sovits/src/start.sh /home/user/GPT-SoVITS

RUN chown -R user /home/user/

USER user
WORKDIR /home/user/GPT-SoVITS

ENTRYPOINT ["bash", "start.sh"]
