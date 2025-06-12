# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ARG ARCH="cpu"  # Set this to "cpu" or "gpu"

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    libgl1-mesa-glx \
    libjemalloc-dev

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu ; \
      $uvpip -r /home/user/comps/web_retrievers/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/web_retrievers/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user

WORKDIR /home/user/comps/web_retrievers/src

ENTRYPOINT ["python", "opea_web_retrievers_microservice.py"]
