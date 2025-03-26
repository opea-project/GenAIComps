#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')


export TAG=comps
export IMAGE2IMAGE_PORT=10401
export service_name="image2image-gaudi"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/image2image-gaudi:$TAG -f comps/image2image/src/Dockerfile.intel_hpu .
    if [ $? -ne 0 ]; then
        echo "opea/image2image built fail"
        exit 1
    else
        echo "opea/image2image built successful"
    fi
}

function start_service() {
    unset http_proxy
    export MODEL='stabilityai/stable-diffusion-xl-refiner-1.0'

    cd $WORKPATH/comps/image2image/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d
    sleep 30s
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:$IMAGE2IMAGE_PORT/v1/image2image -XPOST -d '{"image": "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png", "prompt":"a photo of an astronaut riding a horse on mars", "num_images_per_prompt":1}' -H 'Content-Type: application/json')
    if [[ $result == *"images"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs image2image-gaudi-server
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
