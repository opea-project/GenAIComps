# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ../../FireRedASR ./FireRedASR

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LOGFLAG=true
ENV FIREREDASR_MODEL_DIR=/app/pretrained_models
ENV FIREREDASR_ASR_TYPE=llm

# Expose port
EXPOSE 9099

# Run the microservice
CMD ["python", "src/opea_fireredasr_microservice.py"]
