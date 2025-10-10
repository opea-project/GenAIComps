# Copyright (C) 2025 Zensar Technologies Private Ltd.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/arb_post_hearing_assistant/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/arb_post_hearing_assistant/src/requirements-gpu.txt; \
    fi
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/home/user


RUN mkdir -p /data && chown -R user /data
ENV HF_HUB_CACHE=/data

# air gapped support: pre-download tiktoken bpe files
RUN mkdir -p /opt/tiktoken_cache
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base");tiktoken.get_encoding("o200k_base");tiktoken.get_encoding("gpt2");tiktoken.get_encoding("r50k_base");tiktoken.get_encoding("p50k_base");tiktoken.get_encoding("p50k_edit");'
RUN chown -R user /opt/tiktoken_cache

USER user

WORKDIR /home/user

ENTRYPOINT ["python", "comps/arb_post_hearing_assistant/src/opea_arb_post_hearing_assistant_microservice.py"]

