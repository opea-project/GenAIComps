# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set this to "cpu" or "gpu" or etc
ARG ARCH="cpu"

ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    curl \
    git \
    git-lfs && \
    git lfs install

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user:user /home/user/
RUN mkdir /home/user/model && chown user:user -R /home/user/model

COPY --chown=user:user comps /home/user/comps
WORKDIR /home/user/comps/third_parties/video-llama/src

# install the fixed torch version again after install cpu requirements, make sure the deps are compatible
ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
        $uvpip torch==2.5.1 torchaudio~=2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu; \
        $uvpip -r /home/user/comps/third_parties/video-llama/src/requirements-cpu.txt; \
    else \
        $uvpip -r /home/user/comps/third_parties/video-llama/src/requirements-gpu.txt; \
    fi

ARG VIDEO_LLAMA_REPO=https://github.com/DAMO-NLP-SG/Video-LLaMA.git
ARG VIDEO_LLAMA_COMMIT=0adb19e829254b46a442ee78cdcd609a988c5de7

RUN git clone --depth=1 ${VIDEO_LLAMA_REPO} && \
    cd Video-LLaMA && \
    git fetch --depth=1 origin ${VIDEO_LLAMA_COMMIT} && \
    git checkout ${VIDEO_LLAMA_COMMIT} && \
    git apply --whitespace=fix ../video-llama.patch && \
    mv video_llama ../ && \
    cd .. && \
    rm -rf Video-LLaMA

# Modify the degradations.py file to import rgb_to_grayscale from torchvision.transforms.functional
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' /usr/local/lib/python3.11/site-packages/pytorchvideo/transforms/augmentations.py && \
    sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.v2.functional/' /usr/local/lib/python3.11/site-packages/pytorchvideo/transforms/augmentations.py

USER user

ENV PYTHONPATH=/home/user

ENTRYPOINT ["bash", "start.sh"]
