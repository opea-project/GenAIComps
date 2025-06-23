#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache}

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker build --no-cache -t opea/ipex-llm:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg COMPILE=ON --build-arg PORT_SSH=2345 -f comps/third_parties/ipex/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/ipex-llm built fail"
        exit 1
    else
        echo "opea/ipex-llm built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export host_ip=${ip_address}
    export MODEL_ID="microsoft/phi-4"
    export TAG=comps
    cd $WORKPATH
    cd comps/third_parties/ipex/deployment/docker_compose
    docker compose -f compose.yaml up -d
    echo "Microservice started"
    sleep 120
}

function validate_microservice() {
    echo "Validate microservice started"
    result=$(http_proxy="" curl http://localhost:8688/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/Phi-4-mini-instruct",
    "messages": [
      {"role": "user", "content": "What is Deep Learning?"}
    ],
    "max_tokens": 32
  }'
)
    if [[ $result == *"Deep"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs ipex-llm-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=ipex-llm-server")
    echo "Shutdown legacy containers "$cid
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo "cleanup container images and volumes"
    echo y | docker system prune > /dev/null 2>&1

}

main
