# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8
ARG ARCH=cpu

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    if [ ${ARCH} = "cpu" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /home/user/comps/agent/src/requirements.txt; \
    else \
        pip install --no-cache-dir -r /home/user/comps/agent/src/requirements.txt; \
    fi

ENV PYTHONPATH=/home/user

RUN mkdir -p /home/user/comps/agent/src/status && chown -R user /home/user/comps/agent/src/status

USER user

WORKDIR /home/user/comps/agent/src/

ENTRYPOINT ["python", "agent.py"]
