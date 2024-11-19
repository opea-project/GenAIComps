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


# Create user
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user/comps/chathistory/arango && \
    chown -R user /home/user

USER user

COPY comps /home/user/comps
COPY requirements.txt /home/user/

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/user/comps/chathistory/arango/requirements.txt && \
    pip install --no-cache-dir -r /home/user/requirements.txt   

ENV PYTHONPATH=/home/user/

WORKDIR /home/user/comps/chathistory/arango
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "chathistory_arango.py"]
