# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.11-slim

# Set environment variables
ENV LANG=en_US.UTF-8

ARG ARCH="cpu"

COPY comps /home/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    if [ ${ARCH} = "cpu" ]; then \
      $uvpip torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
      $uvpip -r /home/comps/text2image/src/requirements-cpu.txt; \
    else \
      $uvpip -r /home/comps/text2image/src/requirements-gpu.txt; \
    fi


ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/text2image/src

RUN echo python opea_text2image_microservice.py --bf16 >> run.sh

CMD ["bash", "run.sh"]
