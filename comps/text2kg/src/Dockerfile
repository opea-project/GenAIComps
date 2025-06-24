# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:22.04

WORKDIR /home/graph_extract

FROM python:3.11-slim
ENV LANG=C.UTF-8
ARG ARCH=cpu

RUN apt-get update -y && apt-get install vim wget -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential 

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/ 

COPY comps /home/user/comps

RUN mkdir -p /home/users/comps/text2kg/src/tmpddata

WORKDIR /home/users/comps/text2kg/src/tmpddata

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/text2kg/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/text2kg/src/requirements-gpu.txt; \
    fi

ENV https_proxy=${https_proxy}
ENV http_proxy=${http_proxy}
ENV no_proxy=${no_proxy}
ENV LLM_ID=${LLM_ID:-"HuggingFaceH4/zephyr-7b-alpha"}
ENV SPAN_LENGTH=${SPAN_LENGTH:-"1024"}
ENV OVERLAP=${OVERLAP:-"100"}
ENV MAX_LENGTH=${MAX_NEW_TOKENS:-"256"}
ENV HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
ENV HF_TOKEN=${HF_TOKEN}
ENV TGI_PORT=8008
ENV PYTHONPATH="/home/user/":$PYTHONPATH

USER user

WORKDIR /home/user/comps/text2kg/src/


WORKDIR /home/user/comps/text2kg/src/

RUN bash -c 'source /home/user/comps/text2kg/src/environment_setup.sh'

ENTRYPOINT ["python", "opea_text2kg_microservice.py"]
