# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function build_vllm_docker_images() {
    echo "Building the vllm docker images"
    cd $WORKDIR
    echo $WORKPATH
    if [ ! -d "./vllm" ]; then
        git clone https://github.com/HabanaAI/vllm-fork.git
    fi
    cd ./vllm-fork
    # git fetch --all
    # git checkout v0.6.4.post2+Gaudi-1.19.0
    # sed -i 's/triton/triton==3.1.0/g' requirements-hpu.txt
    docker build --no-cache -f Dockerfile.hpu -t opea/vllm-gaudi:comps --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi:comps failed"
        exit 1
    else
        echo "opea/vllm-gaudi:comps successful"
    fi
}

build_vllm_docker_images
