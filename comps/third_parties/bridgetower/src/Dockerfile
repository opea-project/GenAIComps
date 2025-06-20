# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim

# Set this to "cpu" or "gpu" or etc
ARG ARCH="cpu"

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing curl

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user:/usr/lib/habanalabs/:/optimum-habana

COPY --chown=user comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/third_parties/bridgetower/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/third_parties/bridgetower/src/requirements-gpu.txt; \
    fi



ENV PYTHONPATH=$PYTHONPATH:/home/user
USER user
ARG EMBEDDER_PORT=8080
ENV PORT=$EMBEDDER_PORT

WORKDIR /home/user/comps/third_parties/bridgetower/src/

ENTRYPOINT ["python", "bridgetower_server.py", "--device", "cpu"]
