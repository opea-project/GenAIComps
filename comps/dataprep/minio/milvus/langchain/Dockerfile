# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    default-jre \
    libgl1-mesa-glx \
    libjemalloc-dev \
    tesseract-ocr

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

COPY comps /home/user/comps

RUN pip install --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; fi && \
    pip install -r /home/user/comps/dataprep/minio/milvus/langchain/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER root

RUN mkdir -p /home/user/comps/dataprep/minio/milvus/langchain/uploaded_files && chown -R user /home/user/comps/dataprep/minio/milvus/langchain/uploaded_files

USER user
WORKDIR /home/user/comps/dataprep/minio/milvus/langchain

ENTRYPOINT ["python", "prepare_doc_milvus.py"]
