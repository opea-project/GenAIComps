# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    default-jre \
    libcairo2 \
    libgl1-mesa-glx \
    libjemalloc-dev \
    libmariadb-dev \
    libpq-dev \
    libreoffice \
    poppler-utils \
    tesseract-ocr \
    wget curl

# Install ffmpeg static build
WORKDIR /root
RUN wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
    mkdir ffmpeg-git-amd64-static && tar -xvf ffmpeg-git-amd64-static.tar.xz -C ffmpeg-git-amd64-static --strip-components 1 && \
    export PATH=/root/ffmpeg-git-amd64-static:$PATH && \
    cp /root/ffmpeg-git-amd64-static/ffmpeg /usr/local/bin/


RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip -r /home/user/comps/dataprep/src/requirements-cpu.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match; \
    else \
      $uvpip -r /home/user/comps/dataprep/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user

RUN mkdir -p /home/user/comps/dataprep/src/uploaded_files && chown -R user /home/user/comps/dataprep/src/uploaded_files
RUN mkdir -p /data && chown -R user /data

USER user
ENV NLTK_DATA=/home/user/nltk_data
# air gapped support: predownload all needed nltk data
RUN mkdir -p /home/user/nltk_data && python -m nltk.downloader -d /home/user/nltk_data punkt_tab averaged_perceptron_tagger_eng stopwords
# air gapped support: set model cache dir
ENV HF_HUB_CACHE=/data

WORKDIR /home/user/comps/dataprep/src

ENTRYPOINT ["sh", "-c", "python $( [ \"$MULTIMODAL_DATAPREP\" = \"true\" ] && echo 'opea_dataprep_multimodal_microservice.py' || echo 'opea_dataprep_microservice.py')"]
