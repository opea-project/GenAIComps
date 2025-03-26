#!/bin/bash
# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')  # Adjust to a more reliable command
if [ -z "$host_ip" ]; then
    host_ip="localhost"  # Default to localhost if IP address is empty
fi
LOG_PATH="$WORKPATH/tests"
service_name="textgen-predictionguard"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/llm-textgen:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-textgen built fail"
        exit 1
    else
        echo "opea/llm-textgen built successful"
    fi
}

function start_service() {
    export TEXTGEN_PORT=10513 #10500-10599
    export host_ip=${host_ip}
    export PREDICTIONGUARD_API_KEY=${PREDICTIONGUARD_API_KEY}
    export LOGFLAG=True

    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 60  # Sleep for 1 minute to allow the service to start
}

function validate_microservice() {
    result=$(http_proxy="" curl http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
        -X POST \
        -d '{"model": "Hermes-2-Pro-Llama-3-8B", "messages": "What is AI?", "stream": false, "max_tokens": 100, "temperature": 0.7, "top_p": 1.0, "top_k": 50}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"content"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs ${service_name}
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml down --remove-orphans
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
