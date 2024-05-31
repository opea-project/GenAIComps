FROM vault.habana.ai/gaudi-docker/1.13.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.0:latest as hpu

ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/root:/usr/lib/habanalabs/
ARG REPO=https://github.com/opea-project/GenAIEval.git
ARG REPO_PATH=""
ARG BRANCH=main

RUN apt-get update && \
    apt-get install git-lfs && \
    git-lfs install

# Download code
SHELL ["/bin/bash", "--login", "-c"]
RUN mkdir -p /GenAIEval
COPY ${REPO_PATH} /GenAIEval
RUN if [ "$REPO_PATH" == "" ]; then rm -rf /GenAIEval/* && rm -rf /GenAIEval/.* ; git clone --single-branch --branch=${BRANCH} ${REPO} /GenAIEval ; fi
RUN pip install --upgrade pip setuptools==69.5.1

# Build From Source
RUN cd /GenAIEval && \
    pip install -r requirements.txt && \
    python setup.py install && \
    pip install --upgrade-strategy eager optimum[habana] && \
    pip list

WORKDIR /GenAIEval/