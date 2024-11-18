# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libjemalloc-dev \
    libgl1-mesa-glx

# Install Python packages globally
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-arango==7.9.1 \
    pydantic==2.5.2

# Create user
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user/comps/chathistory/arangodb && \
    chown -R user /home/user

USER user

ENV PYTHONPATH=/home/user/comps

WORKDIR /home/user/comps/chathistory/arangodb

ENTRYPOINT ["python", "chathistory_arango.py"]
