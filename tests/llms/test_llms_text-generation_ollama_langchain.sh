#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    ollama pull llama3
    docker build --no-cache \
        -t opea/llm-ollama:latest \
        --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy \
        -f comps/llms/text-generation/ollama/langchain/Dockerfile .

    if [ $? -ne 0 ]; then
        echo "opea/llm-ollama built fail"
        exit 1
    else
        echo "opea/llm-ollama built successful"
    fi
}

function start_service() {
    docker run -d \
     --name="test-comps-llm-ollama-server" \
     --network host \
     -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
     opea/llm-ollama:latest

}

function validate_microservice() {
    result=$(http_proxy="" curl http://${ip_address}:9000/v1/chat/completions \
        -X POST \
        -d '{"model": "llama3", "query":"What is Deep Learning?","max_new_tokens":32,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":false}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"text"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-llm-ollama-server
        exit 1
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-llm-ollama*")
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
