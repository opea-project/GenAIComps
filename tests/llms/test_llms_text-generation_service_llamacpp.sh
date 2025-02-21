#!/bin/bash
# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD") # Assumes the script is called from GenAIComps/comps
host_ip=$(hostname -I | awk '{print $1}')  # Adjust to a more reliable command
if [ -z "$host_ip" ]; then
    host_ip="localhost"  # Default to localhost if IP address is empty
fi
LOG_PATH="$WORKPATH/tests"
service_name="textgen-service-llamacpp"


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
    export LLM_ENDPOINT_PORT=8008
    export LLM_ENDPOINT="http://${host_ip}:80"
    export TEXTGEN_PORT=9000
    export LLM_MODEL_ID="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    export LLAMA_ARG_CTX_SIZE=4096
    export LOGFLAG=True

    export MODEL_PATH=~/models
    mkdir -p $MODEL_PATH
    cd $MODELPATH
    wget --no-clobber https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose_llama.log
    sleep 120  # Allow the service to start
}

function validate_microservice() {
    result=$(http_proxy="" curl -X POST http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "What is AI?"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": false
    }')

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
    docker compose -f compose_text-generation.yaml down ${service_name} --remove-orphans
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
set +x
