#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')


export TAG=comps
export IMAGE2IMAGE_PORT=10400
export service_name="image2image"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/image2image:$TAG -f comps/image2image/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/image2image built fail"
        exit 1
    else
        echo "opea/image2image built successful"
    fi
}

function start_service() {
    unset http_proxy
    cd $WORKPATH/comps/image2image/deployment/docker_compose
    export MODEL='stabilityai/stable-diffusion-xl-refiner-1.0'
    export ENABLE_MCP=True
    docker compose -f compose.yaml up ${service_name} -d
    sleep 30s
}

function validate_microservice() {
    python3 ${WORKPATH}/tests/utils/validate_svc_with_mcp.py $ip_address $IMAGE2IMAGE_PORT "image2image"
    if [ $? -ne 0 ]; then
        echo "Result wrong."
        docker logs image2image-server
        exit 1
    fi

}

function stop_docker() {
    cd $WORKPATH/comps/image2image/deployment/docker_compose
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
