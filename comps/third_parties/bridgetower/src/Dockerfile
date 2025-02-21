# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/
RUN apt-get update && apt-get install -y curl

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user:/usr/lib/habanalabs/:/optimum-habana

COPY --chown=user comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/user/comps/third_parties/bridgetower/src/requirements.txt
    

ENV PYTHONPATH=$PYTHONPATH:/home/user
USER user
ARG EMBEDDER_PORT=8080
ENV PORT=$EMBEDDER_PORT

WORKDIR /home/user/comps/third_parties/bridgetower/src/

ENTRYPOINT ["python", "bridgetower_server.py", "--device", "cpu"]
