# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:22.04

WORKDIR /home/graph_extract

FROM python:3.11-slim
ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install vim -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/user/comps/struct2graph/src/requirements.txt

ENV https_proxy=${https_proxy}
ENV http_proxy=${http_proxy}
ENV no_proxy=${no_proxy}
ENV PYTHONPATH="/home/user/":$PYTHONPATH

USER user

WORKDIR /home/user/comps/struct2graph/src/

ENTRYPOINT ["python", "opea_struct2graph_microservice.py"]
