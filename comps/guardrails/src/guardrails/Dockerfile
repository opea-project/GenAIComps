# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/guardrails/src/guardrails/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/guardrails/src/guardrails/requirements-gpu.txt; \
    fi


ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user

WORKDIR /home/user/comps/guardrails/src/guardrails/

ENTRYPOINT ["python", "opea_guardrails_microservice.py"]
