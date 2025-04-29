#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache}
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker build --no-cache -t opea/sglang:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/sglang/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/sglang built fail"
        exit 1
    else
        echo "opea/sglang built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export host_ip=${ip_address}
    export MODEL_ID=${MODEL_ID}
    export TAG=comps
    cd $WORKPATH
    cd comps/third_parties/sglang/deployment/docker_compose
    docker compose -f compose.yaml up -d
    echo "Microservice started"
    sleep 40m
}

function validate_microservice() {
    echo "Validate microservice started"
    result=$(http_proxy="" curl http://localhost:8699/v1/chat/completions \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": ${MODEL_ID},
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
        docker logs sglang-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=sglang-server")
    echo "Shutdown legacy containers "$cid
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {
    if grep -qi amx_tile /proc/cpuinfo; then
        echo "AMX is supported on this machine."
    else
        echo "AMX is NOT supported on this machine, skip this test."
        exit 0
    fi
    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo "cleanup container images and volumes"
    echo y | docker system prune 2>&1 > /dev/null

}

main
