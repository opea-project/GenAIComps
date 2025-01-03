# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    libgl1-mesa-glx \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

# Assumes we're building from the GenAIComps directory.
COPY ../../../comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/user/comps/llms/text-generation/llamacpp/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

WORKDIR /home/user/comps/llms/text-generation/llamacpp/

ENTRYPOINT ["bash", "entrypoint.sh"]
