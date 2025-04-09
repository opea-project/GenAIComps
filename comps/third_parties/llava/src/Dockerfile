# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

# Install system dependencies
RUN apt-get update \
    && apt-get install -y curl

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user:/usr/lib/habanalabs/:/optimum-habana

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/user/comps/third_parties/llava/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user
WORKDIR /home/user/comps/third_parties/llava/src

ENTRYPOINT ["python", "llava_server.py", "--device", "cpu"]
