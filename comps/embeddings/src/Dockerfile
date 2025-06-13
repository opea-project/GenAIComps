# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

COPY comps /home/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/comps/embeddings/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/embeddings/src/

ENV MULTIMODAL_EMBEDDING="false"

ENTRYPOINT ["sh", "-c", "python $( [ \"$MULTIMODAL_EMBEDDING\" = \"true\" ] && echo 'opea_multimodal_embedding_microservice.py' || echo 'opea_embedding_microservice.py')"]

