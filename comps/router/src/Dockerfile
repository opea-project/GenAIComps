# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.10-slim

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Add a non-root user
RUN useradd -m -s /bin/bash user && chown -R user /home/user

# Copy the *entire* comps/ package
WORKDIR /home/user
COPY comps /home/user/comps

# Install deps from the router’s requirements.txt
ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip uv && \
    $uvpip -r /home/user/comps/router/src/requirements.txt && \
    $uvpip torch --index-url https://download.pytorch.org/whl/cpu && \
    git clone --depth 1 https://github.com/lm-sys/RouteLLM.git /tmp/RouteLLM && \
    patch -p1 -d /tmp/RouteLLM < /home/user/comps/router/src/hf_compatibility.patch && \
    $uvpip /tmp/RouteLLM  && rm -rf /tmp/RouteLLM

# Make imports work
ENV PYTHONPATH=/home/user

# Switch to non-root
USER user

# Expose the port
EXPOSE 6000

# Run the microservice
WORKDIR /home/user/comps/router/src
CMD ["python", "opea_router_microservice.py"]
