#!/bin/bash
# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identifier: Apache-2.0

set -x  # Print commands and their arguments as they are executed

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')  # Adjust to a more reliable command
if [ -z "$ip_address" ]; then
    ip_address="localhost"  # Default to localhost if IP address is empty
fi

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/llm-pg:comps -f comps/llms/summarization/predictionguard/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-pg built failed"
        exit 1
    else
        echo "opea/llm-pg built successfully"
    fi
}

function start_service() {
    llm_service_port=9000
    unset http_proxy
    docker run -d --name=test-comps-llm-pg-server \
        -e http_proxy= -e https_proxy= \
        -e PREDICTIONGUARD_API_KEY=${PREDICTIONGUARD_API_KEY} \
        -p 9000:9000 --ipc=host opea/llm-pg:comps
    sleep 60  # Sleep for 5 seconds to allow the service to start
}

function validate_microservice() {
    llm_service_port=9000
    result=$(http_proxy="" curl http://${ip_address}:${llm_service_port}/v1/chat/docsum \
        -X POST \
        -d '{"model": "Hermes-3-Llama-3.1-8B", "query": "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data.", "streaming": false, "max_tokens": 100, "temperature": 0.7, "top_p": 1.0, "top_k": 50}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"text"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-llm-pg-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-llm-pg-*")
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
