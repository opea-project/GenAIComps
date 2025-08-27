# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

# Set environment variables
ENV LANG=en_US.UTF-8
ARG ARCH=cpu

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    ffmpeg

COPY --chown=user:user comps /home/user/comps

RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /home/user/comps/third_parties/funasr/src/requirements-cpu.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user
USER user
WORKDIR /home/user/comps/third_parties/funasr/src

ENTRYPOINT ["python", "funasr_server.py", "--device", "cpu"]
