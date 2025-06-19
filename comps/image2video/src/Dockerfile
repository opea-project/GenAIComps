# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8

ARG ARCH="cpu"

COPY comps /home/comps

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing python3-opencv

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/comps/image2video/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/comps/image2video/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home

USER user

WORKDIR /home/comps/image2video/src

ENTRYPOINT ["python", "opea_image2video_microservice.py"]
