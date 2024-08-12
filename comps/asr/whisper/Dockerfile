# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8 \
    PYTHONPATH=/home/user
ARG ARCH=cpu

# Install system dependencies
RUN apt-get update \
    && apt-get install -y ffmpeg

COPY comps /home/comps

RUN pip install --no-cache-dir --upgrade pip && \
    if [ "${ARCH}" = "cpu" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /home/comps/asr/requirements.txt ; \
    else \
        pip install --no-cache-dir -r /home/comps/asr/requirements.txt ; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/asr/whisper

ENTRYPOINT ["python", "whisper_server.py", "--device", "cpu"]
