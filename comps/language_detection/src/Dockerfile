# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8

RUN apt-get update -y && \
    apt-get install build-essential -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libjemalloc-dev \
    vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY comps /home/comps

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/comps/language_detection/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home

USER user

WORKDIR /home/comps/language_detection/src

ENTRYPOINT ["python", "opea_language_detection_microservice.py"]
