#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
service_name="text2image"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2image:latest -f comps/text2image/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2image built fail"
        exit 1
    else
        echo "opea/text2image built successful"
    fi
}

function start_service() {
    unset http_proxy
    export MODEL=stabilityai/stable-diffusion-xl-base-1.0
    cd $WORKPATH/comps/text2image/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 30s
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:9379/v1/text2image -XPOST -d '{"prompt":"An astronaut riding a green horse", "num_images_per_prompt":1}' -H 'Content-Type: application/json')
    if [[ $result == *"images"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs text2image
        exit 1
    fi

}

function stop_docker() {
    cd $WORKPATH/comps/text2image/deployment/docker_compose
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
