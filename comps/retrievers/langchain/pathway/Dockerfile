# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM pathwaycom/pathway:0.13.2-slim

ENV DOCKER_BUILDKIT=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    poppler-utils \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install -U pathway[xpack-llm]
RUN pip install "unstructured[all-docs] >= 0.10.28, < 0.15"
RUN pip install langchain_openai
RUN pip install sentence_transformers

COPY pathway_vs.py /app/


CMD ["python", "pathway_vs.py"]

