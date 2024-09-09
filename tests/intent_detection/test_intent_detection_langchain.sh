#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/llm-tgi:latest -f comps/intent_detection/langchain/Dockerfile .
}

function start_service() {
    tgi_endpoint=5044
    # Remember to set HF_TOKEN before invoking this test!
    export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
    model=Intel/neural-chat-7b-v3-3
    docker run -d --name="test-comps-intent-tgi-endpoint" -p $tgi_endpoint:80 -v ./data:/data --shm-size 1g ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model

    export TGI_LLM_ENDPOINT="http://${ip_address}:${tgi_endpoint}"
    intent_port=5043
    unset http_proxy
    docker run -d --name="test-comps-intent-server" -p ${intent_port}:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TGI_LLM_ENDPOINT=$TGI_LLM_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN opea/llm-tgi:latest
    sleep 5m
}

function validate_microservice() {
    intent_port=5043
    result=$(http_proxy="" curl http://${ip_address}:${intent_port}/v1/chat/intent\
        -X POST \
        -d '{"query":"What is Deep Learning?","max_new_tokens":10,"top_k":1,"temperature":0.001,"streaming":false}' \
        -H 'Content-Type: application/json')

    echo "==============="
    echo $result
    
    docker logs test-comps-intent-server >> ${LOG_PATH}/intent_detection.log
    docker logs test-comps-intent-tgi-endpoint >> ${LOG_PATH}/tgi-endpoint.log
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-intent*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
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
