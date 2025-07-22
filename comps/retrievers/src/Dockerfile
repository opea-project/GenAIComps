# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ARG ARCH="cpu"

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    libcairo2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjemalloc-dev \
    libmariadb-dev \
    build-essential \
    curl

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
      $uvpip -r /home/user/comps/retrievers/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/user/comps/retrievers/src/requirements-gpu.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user
ENV NLTK_DATA=/home/user/nltk_data
# air gapped support: predownload all needed nltk data
RUN mkdir -p /home/user/nltk_data && python -m nltk.downloader -d /home/user/nltk_data punkt_tab stopwords

WORKDIR /home/user/comps/retrievers/src

ENTRYPOINT ["python", "opea_retrievers_microservice.py"]
