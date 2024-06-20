#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -eu

default_port=8080
default_card_num=0
default_model_cache_directory="${HOME}/.cache/huggingface/hub"
HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}

docker_cmd=<<EOF 
docker run -d \
    --name=TritonStabilityServer -p ${port_number}:8000 \
    -e HABANA_VISIBLE_DEVICES=${default_card_num} \
    -e HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN} \
    --cap-add=sys_nic \
    --ipc=host \
    --runtime=habana \
    -v ${default_model_cache_directory}:/root/.cache/huggingface/hub \
    ohio-stability-triton
EOF

eval $docker_cmd
