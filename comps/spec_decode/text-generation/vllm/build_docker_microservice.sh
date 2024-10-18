# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cd ../../../../
docker build  \
    -t opea/spec_decode-vllm:latest \
    --build-arg https_proxy=$https_proxy \
    --build-arg http_proxy=$http_proxy \
    -f comps/spec_decode/text-generation/vllm/docker/Dockerfile.microservice .
