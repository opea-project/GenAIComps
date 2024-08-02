# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.9-slim

ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    git git-lfs vim && \
    git lfs install

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user:user /home/user/

USER user

#comment out this, only copy requirements.txt to avoid replicate pip install when files under comps change
# COPY --chown=user:user comps /home/user/comps
# WORKDIR /home/user/comps/lvms/video-llama/server
COPY --chown=user:user comps/lvms/video-llama/server/requirements.txt /home/user/comps/lvms/video-llama/server/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /home/user/comps/lvms/video-llama/server/requirements.txt

# FIXME: remove this and add to requirement when done
RUN pip install sentence-transformers==3.0.1
RUN pip install faiss-cpu
RUN pip install sentencepiece

# FIXME: remove this when done
COPY --chown=user:user comps /home/user/comps
WORKDIR /home/user/comps/lvms/video-llama/server

ARG VIDEO_LLAMA_REPO=https://github.com/DAMO-NLP-SG/Video-LLaMA.git
ARG VIDEO_LLAMA_COMMIT=0adb19e
RUN git clone ${VIDEO_LLAMA_REPO} Video-LLaMA && \
    cd Video-LLaMA && git checkout ${VIDEO_LLAMA_COMMIT} && \
    git apply --whitespace=fix ../video-llama.patch && \
    mv video_llama ../ && \
    cd ../ && rm -rf Video-LLaMA


ENV PYTHONPATH=/home/user

# FIXME: move to line 15
USER root
RUN mkdir /home/user/model && chown user:user -R /home/user/model
USER user

ENTRYPOINT ["bash", "start.sh"]