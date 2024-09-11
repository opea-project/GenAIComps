# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cd ../../../../

docker build \
<<<<<<<< HEAD:comps/llms/text-generation/vllm-ray/build_docker_vllmray.sh
    -f comps/llms/text-generation/vllm-ray/docker/Dockerfile.vllmray \
    -t opea/vllm_ray:habana \
========
    -f comps/llms/text-generation/ray_serve/docker/Dockerfile.rayserve \
    -t ray_serve:habana \
>>>>>>>> origin/main:comps/llms/text-generation/ray_serve/build_docker_rayserve.sh
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} .
