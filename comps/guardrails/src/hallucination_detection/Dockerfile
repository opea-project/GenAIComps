# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ARG ARCH="cpu"  # Set this to "cpu" or "gpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libgl1-mesa-glx \
    libjemalloc-dev

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      uv pip install --no-cache-dir -r /home/user/comps/guardrails/src/hallucination_detection/requirements-cpu.txt; \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match; \
    else \
      uv pip install --no-cache-dir -r /home/user/comps/guardrails/src/hallucination_detection/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user

WORKDIR /home/user/comps/guardrails/src/hallucination_detection

ENTRYPOINT ["python", "opea_hallucination_detection_microservice.py"]
