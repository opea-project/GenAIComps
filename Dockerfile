# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Base image for GenAIComps based OPEA Python applications
# Build: docker build -t opea/comps-base -f Dockerfile .

ARG IMAGE_NAME=python
ARG IMAGE_TAG=3.11-slim

FROM ${IMAGE_NAME}:${IMAGE_TAG} AS base

ENV HOME=/home/user

RUN useradd -m -s /bin/bash user && \
    mkdir -p $HOME && \
    chown -R user $HOME

# get security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR $HOME

COPY *.toml *.py *.txt *.md LICENSE ./

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY comps/ comps/

ENV PYTHONPATH=$PYTHONPATH:$HOME

USER user

ENTRYPOINT ["sh", "-c", "set && ls -la"]
