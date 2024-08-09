# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

From ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

ENV GLIBC_TUNABLES glibc.cpu.x86_shstk=permissive

COPY comps /root/comps
COPY neural-speed-0.1-cp310-cp310-linux_x86_64.whl /root/
COPY bert-q8j-g-1-cint8.bin /root/

RUN apt update && apt install -y python3 python3-pip 
RUN pip3 install -r /root/comps/embeddings/neural-speed/neuralspeed-docker/requirements.txt
RUN pip3 install llmspec mosec msgspec httpx requests
RUN pip3 install /root/neural-speed-0.1-cp310-cp310-linux_x86_64.whl

RUN cd /root/ && export HF_ENDPOINT=https://hf-mirror.com && huggingface-cli download --resume-download maidalun1020/bce-embedding-base_v1 --local-dir /root/bce-embedding-base_v1

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

WORKDIR /root/comps/embeddings/neural-speed/neuralspeed-docker

CMD ["python3", "server.py"]
