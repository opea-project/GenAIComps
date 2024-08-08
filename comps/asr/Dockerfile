# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

ENV LANG=C.UTF-8
ARG ARCH=cpu

COPY comps /home/comps

RUN pip install --no-cache-dir --upgrade pip && \
    if [ ${ARCH} = "cpu"]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /home/comps/asr/requirements.txt && \
        pip freeze | grep nvidia | xargs pip uninstall -y ; \
    else \
        pip install --no-cache-dir -r /home/comps/asr/requirements.txt; \
    fi

ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/asr

ENTRYPOINT ["python", "asr.py"]