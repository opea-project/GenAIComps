# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"
ARG SERVICE="default"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

COPY comps /home/user/comps

RUN if [ ${SERVICE} = "fastRAG" ]; then \
        git clone https://github.com/IntelLabs/fastRAG.git /home/user/fastRAG && \
        cd /home/user/fastRAG && \
        pip install --no-cache-dir --upgrade pip && \
        if [ ${ARCH} = "cpu" ]; then \
            pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu; \
            pip install --no-cache-dir .; \
            pip install --no-cache-dir .[intel]; \
        else \
            pip install --no-cache-dir .; \
            pip install --no-cache-dir .[intel]; \
        fi && \
        pip install --no-cache-dir -r /home/user/comps/reranks/src/requirements_fastRAG.txt; \
    else \
        pip install --no-cache-dir --upgrade pip setuptools && \
        if [ ${ARCH} = "cpu" ]; then \
            pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
        fi; \
    fi && \
    pip install --no-cache-dir -r /home/user/comps/reranks/src/requirements.txt;


ENV PYTHONPATH=$PYTHONPATH:/home/user

WORKDIR /home/user/comps/reranks/src

ENTRYPOINT ["python", "opea_reranking_microservice.py"]
