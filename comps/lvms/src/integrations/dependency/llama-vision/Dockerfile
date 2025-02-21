# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.19.0/ubuntu22.04/habanalabs/pytorch-installer-2.5.1:latest AS hpu

ENV LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    git-lfs \
    libgl1-mesa-glx \
    libjemalloc-dev \
    curl

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

RUN git lfs install

COPY comps /home/user/comps

RUN cd /home/user/comps/lvms/src/integrations/dependency/llama-vision/ && \
    pip install --no-cache-dir -r requirements.txt  && \
    pip install --no-cache-dir --upgrade Pillow

RUN pip install --no-cache-dir git+https://github.com/huggingface/optimum-habana@v1.15.0

ENV PYTHONPATH=/root:/home/user

USER user

WORKDIR /home/user/comps/lvms/src/integrations/dependency/llama-vision/

ENTRYPOINT ["python", "lvm.py"]
