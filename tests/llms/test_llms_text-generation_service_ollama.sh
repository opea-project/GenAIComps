#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="textgen-service-ollama"

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
    export LLM_ENDPOINT_PORT=12114  # 12100-12199
    export TEXTGEN_PORT=10514 #10500-10599
    export host_ip=${host_ip}
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export LLM_MODEL_ID=$1
    export LOGFLAG=True

    cd $WORKPATH/comps/third_parties/ollama/deployment/docker_compose/
    docker compose -f compose.yaml up -d > ${LOG_PATH}/start_services_with_compose_ollama.log

    sleep 5s
    docker exec ollama-server ollama pull $LLM_MODEL_ID
    sleep 20s

    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 30s
}

function validate_microservice() {
    result=$(http_proxy="" curl http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
        -X POST \
        -d '{"messages": [{"role": "user", "content": "What is Deep Learning?"}]}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"content"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs ollama-server >> ${LOG_PATH}/llm-ollama.log
        docker logs ${service_name} >> ${LOG_PATH}/llm-server.log
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

    llm_models=(
    llama3.2:1b
    )
    for model in "${llm_models[@]}"; do
      start_service "${model}"
      validate_microservice
      stop_docker
    done

    echo y | docker system prune

}

main
