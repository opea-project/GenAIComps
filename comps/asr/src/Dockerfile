# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

ENV LANG=C.UTF-8
ARG ARCH=cpu

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/asr/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/asr/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user
USER user
WORKDIR /home/user/comps/asr/src

ENTRYPOINT ["python", "opea_asr_microservice.py"]
