# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8

COPY comps /home/user/comps

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing curl

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/user/comps/third_parties/predictionguard/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

WORKDIR /home/user/comps/third_parties/predictionguard/src

ENTRYPOINT ["python", "lvm.py"]
