
# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identitier: Apache-2.0

FROM python:3.11-slim

COPY comps /home/comps

ARG uvpip='uv pip install --system --no-cache-dir'
RUN pip install --no-cache-dir --upgrade pip setuptools uv && \
    $uvpip -r /home/comps/guardrails/src/pii_detection/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home

WORKDIR /home/comps/guardrails/src/pii_detection/

ENTRYPOINT ["python", "opea_pii_detection_microservice.py" ]
