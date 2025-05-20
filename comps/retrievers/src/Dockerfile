# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libcairo2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjemalloc-dev \
    libmariadb-dev \
    build-essential

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then \
        PIP_EXTRA_INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cpu"; \
    else \
        PIP_EXTRA_INDEX_URL=""; \
    fi && \
    pip install --no-cache-dir torch torchvision ${PIP_EXTRA_INDEX_URL} && \
    pip install --no-cache-dir ${PIP_EXTRA_INDEX_URL} -r /home/user/comps/retrievers/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user
ENV NLTK_DATA=/home/user/nltk_data

WORKDIR /home/user/comps/retrievers/src

ENTRYPOINT ["python", "opea_retrievers_microservice.py"]
