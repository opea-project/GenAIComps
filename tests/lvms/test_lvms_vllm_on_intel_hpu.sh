#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export VLLM_PORT=11510
export LVM_PORT=11511
export LLM_MODEL_ID="llava-hf/llava-1.5-7b-hf"
export HUGGINGFACEHUB_API_TOKEN=$HF_TOKEN
export VLLM_SKIP_WARMUP=true

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    git clone https://github.com/HabanaAI/vllm-fork.git
    cd ./vllm-fork/
    git checkout habana_main
    docker build --no-cache -f Dockerfile.hpu -t opea/vllm-gaudi:$TAG --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
    cd ..
    rm -rf vllm-fork

    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm:$TAG -f comps/lvms/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/lvm built fail"
        exit 1
    else
        echo "opea/lvm built successful"
    fi
}

function start_service() {

    export LVM_ENDPOINT=http://$ip_address:$VLLM_PORT

    export LVM_COMPONENT_NAME=OPEA_VLLM_LVM
    docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up lvm-vllm-gaudi vllm-gaudi-service -d
    sleep 15s
}

function validate_microservice() {

    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs vllm-gaudi-service >> ${LOG_PATH}/vllm-dependency.log
        docker logs lvm-vllm-gaudi-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi

    # Test the LVM with text only (no image)
    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": "", "prompt":"What is deep learning?"}' -H 'Content-Type: application/json')
    if [[ $result == *"Deep learning is"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs vllm-gaudi-service >> ${LOG_PATH}/vllm-dependency.log
        docker logs lvm-vllm-gaudi-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=vllm-gaudi-service" --filter "name=lvm-vllm-gaudi-service" --format "{{.Names}}" | xargs -r docker stop
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
