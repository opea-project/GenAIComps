# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/
RUN apt-get update && apt-get install -y curl
USER user
# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user:/usr/lib/habanalabs/:/optimum-habana

COPY --chown=user comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /home/user/comps/3rd_parties/bridgetower/deployment/docker_compose/requirements.txt
    

ENV PYTHONPATH=$PYTHONPATH:/home/user

ARG EMBEDDER_PORT=8080
ENV PORT=$EMBEDDER_PORT

WORKDIR /home/user/comps/comps/3rd_parties/bridgetower/deployment/docker_compose/

ENTRYPOINT ["python", "bridgetower_server.py", "--device", "cpu"]
