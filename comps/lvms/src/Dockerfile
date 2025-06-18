# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set this to "cpu" or "gpu" or etc
ARG ARCH="cpu"

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

ENV LANG=C.UTF-8

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/lvms/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/lvms/src/requirements-gpu.txt; \
    fi && \
    $uvpip --upgrade transformers

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user

WORKDIR /home/user/comps/lvms/src

ENTRYPOINT ["python", "opea_lvm_microservice.py"]
