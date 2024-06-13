# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# HABANA environment
FROM vault.habana.ai/gaudi-docker/1.14.0/ubuntu22.04/habanalabs/pytorch-installer-2.1.1 AS hpu
RUN rm -rf /etc/ssh/ssh_host*

# Set environment variables
ENV LANG=en_US.UTF-8
ENV PYTHONPATH=/home/user:/usr/lib/habanalabs/:/optimum-habana

# Install required branch
# RUN git clone https://github.com/lkk12014402/optimum-habana.git /optimum-habana -b enable_llava_generation

COPY comps /home/comps

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /home/comps/img2txt/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/img2txt

ENTRYPOINT ["python", "img2txt.py"]