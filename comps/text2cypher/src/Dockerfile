# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG UBUNTU_VERSION=22.04
ARG SYNAPSEAI_VERSION=1.17.0
ARG INSTALLER_VERSION=2.3.1

FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8
ARG PYTHON=python3

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PYTHON} -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools \
    psutil

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

#RUN curl -fsSL https://ollama.com/install.sh | sh

# Install tests
#RUN ${PYTHON} -m pip install pytest
#RUN ${PYTHON} -m pip install ipdb

# Install requirements
RUN ${PYTHON} -m pip install neo4j 
RUN ${PYTHON} -m pip install deepspeed 
RUN ${PYTHON} -m pip install Accelerate
RUN ${PYTHON} -m pip install pyprojroot 

# Lumped all packages to get better dependency evaluation
RUN ${PYTHON} -m pip install --upgrade-strategy eager optimum[habana] \
                             llama-index \
                             llama-index-core \
                             llama-index-graph_stores-neo4j \
                             llama-index-llms-huggingface \
                             llama-index-embeddings-huggingface  \
                             llama-index-llms-huggingface \
                             llama-index-llms-huggingface-api \
                             langchain-community \
                             langchain-huggingface \
                             gradio

# Add the ubuntu user
RUN useradd -m ubuntu
RUN echo 'ubuntu ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu
