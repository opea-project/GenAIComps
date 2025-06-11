# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM pathwaycom/pathway:0.13.2-slim

ENV DOCKER_BUILDKIT=1
ENV PYTHONUNBUFFERED=1
ARG ARCH="cpu"

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    libmagic-dev \
    libreoffice \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY comps/third_parties/pathway/src/requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      uv pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu;
      uv pip install --no-cache-dir -r requirements-cpu.txt; \
    else \
      uv pip install --no-cache-dir -r requirements-gpu.txt; \
    fi

COPY comps/third_parties/pathway/src/vectorstore_pathway.py /app/


CMD ["python", "vectorstore_pathway.py"]

