#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')


export TAG=comps
export PORT=8069
export service_name="language-detection"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/language-detection:$TAG -f comps/language_detection/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/language-detection built fail"
        exit 1
    else
        echo "opea/language-detection built successful"
    fi
}

function start_service() {
    unset http_proxy
    cd $WORKPATH/comps/language_detection/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d
    sleep 3s
}

function validate_microservice() {
    result=$(http_proxy="" curl -X POST -H "Content-Type: application/json" -d @- http://localhost:$PORT/v1/language_detection <<JSON_DATA
{
  "text": "Hi. I am doing fine.",
  "prompt": "### You are a helpful, respectful, and honest assistant to help the user with questions. \
Please refer to the search results obtained from the local knowledge base. \
But be careful to not incorporate information that you think is not relevant to the question. \
If you don't know the answer to a question, please don't share false information. \
### Search results:   \n
### Question: 你好。你好吗？ \n
### Answer:"
}
JSON_DATA
)
    if [[ $result == *"\"source_lang\":\"English\",\"target_lang\":\"Chinese\""* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs language-detection
        exit 1
    fi

}

function stop_docker() {
    cd $WORKPATH/comps/language_detection/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
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
