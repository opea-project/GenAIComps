# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function build_vllm_docker_images() {
    echo "Building the vllm docker images"
    cd $WORKDIR
    echo $WORKPATH
    source $(git rev-parse --show-toplevel)/.github/env/_vllm_versions.sh
    git clone --depth 1 -b ${VLLM_FORK_VER} --single-branch https://github.com/HabanaAI/vllm-fork.git && cd ./vllm-fork
    docker build --no-cache -f Dockerfile.hpu -t opea/vllm-gaudi:comps --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi:comps failed"
        exit 1
    else
        echo "opea/vllm-gaudi:comps successful"
    fi
}

build_vllm_docker_images
