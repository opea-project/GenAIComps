# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    default-jre \
    libcairo2 \
    libjemalloc-dev \
    libmariadb-dev \
    libpq-dev \
    libreoffice \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    wget curl

# Add Microsoft package repo
RUN DEBIAN_VERSION=$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2 | cut -d '.' -f 1) && \
    curl -sSL -O https://packages.microsoft.com/config/debian/$DEBIAN_VERSION/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb

# Install Microsoft ODBC Driver and tools
RUN apt-get update && \
    ACCEPT_EULA=Y apt-get install -y \
    msodbcsql18 \
    mssql-tools18 \
    unixodbc-dev \
    libgssapi-krb5-2 && \
    echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/user/comps/dataprep/src/requirements-cpu.txt; \
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
