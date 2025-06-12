# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim
RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

ENV LANG=C.UTF-8

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/user/comps/tts/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user
USER user
WORKDIR /home/user/comps/tts/src

ENTRYPOINT ["python", "opea_tts_microservice.py"]
