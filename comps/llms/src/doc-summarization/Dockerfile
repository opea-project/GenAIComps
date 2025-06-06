# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi && \
    pip install --no-cache-dir -r /home/user/comps/llms/src/doc-summarization/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

# air gapped support: set model cache dir
RUN mkdir -p /data && chown -R user /data
ENV HF_HUB_CACHE=/data
# air gapped support: pre-download tiktoken bpe files
RUN mkdir -p /opt/tiktoken_cache
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base");tiktoken.get_encoding("o200k_base");tiktoken.get_encoding("gpt2");tiktoken.get_encoding("r50k_base");tiktoken.get_encoding("p50k_base");tiktoken.get_encoding("p50k_edit");'
RUN chown -R user /opt/tiktoken_cache

USER user

WORKDIR /home/user/comps/llms/src/doc-summarization

ENTRYPOINT ["python", "opea_docsum_microservice.py"]
