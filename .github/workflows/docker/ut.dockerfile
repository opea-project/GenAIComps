# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG UBUNTU_VER=24.04
FROM ubuntu:${UBUNTU_VER} as devel

ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    aspell \
    aspell-en \
    build-essential \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget

RUN ln -sf $(which python3) /usr/bin/python
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN python -m pip install --no-cache-dir pytest pytest-cov uv

WORKDIR /
