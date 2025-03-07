# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:22.04

WORKDIR /home/graph_extract

FROM python:3.11-slim
ENV LANG=C.UTF-8
ARG ARCH=cpu

RUN apt-get update -y && apt-get install vim -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential 

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /home/user/comps/text2graph/src/requirements.txt; \
    else \
        pip install --no-cache-dir -r /home/user/comps/text2graph/src/requirements.txt; \
    fi

ENV https_proxy=${https_proxy}
ENV http_proxy=${http_proxy}
ENV no_proxy=${no_proxy}
ENV LLM_ID=${LLM_ID:-"Babelscape/rebel-large"}
ENV SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
ENV OVERLAP=${OVERLAP:-"100"}
ENV MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
ENV HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
ENV HF_TOKEN=${HF_TOKEN}
ENV LLM_MODEL_ID=${LLM_ID}
ENV TGI_PORT=8008
ENV PYTHONPATH="/home/user/":$PYTHONPATH

USER user

WORKDIR /home/user/comps/text2graph/src/

RUN bash -c 'source /home/user/comps/text2graph/src/setup_service_env.sh'

ENTRYPOINT ["python", "opea_text2graph_microservice.py"]
