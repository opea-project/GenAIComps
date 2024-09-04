#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')  # Adjust to a more reliable command

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding-pg:comps -f comps/embeddings/predictionguard/docker/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/embedding-pg built fail"
        exit 1
    else
        echo "opea/embedding-pg built successfully"
    fi
}

function start_service() {
    tei_service_port=6000
    unset http_proxy
    # Set your API key here (ensure this environment variable is set)
    export PREDICTIONGUARD_API_KEY="your_actual_api_key"
    docker run -d --name=test-comps-embedding-pg-server \
    -e http_proxy= -e https_proxy= \
    -e PREDICTIONGUARD_API_KEY=${PREDICTIONGUARD_API_KEY} \
    -p 6000:6000 --ipc=host opea/embedding-pg:comps
    sleep 180
}

function validate_microservice() {
    tei_service_port=6000
    result=$(http_proxy="" curl http://${ip_address}:$tei_service_port/v1/embeddings \
        -X POST \
        -d '{"text":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-embedding-pg-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-embedding-pg-*")
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
