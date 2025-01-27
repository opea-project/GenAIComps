#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
service_name="image2video"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/image2video:latest -f comps/image2video/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/image2video built fail"
        exit 1
    else
        echo "opea/image2video built successful"
    fi
}

function start_service() {
    unset http_proxy
    cd $WORKPATH/comps/image2video/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 3m
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:9369/v1/image2video -XPOST -d '{"images_path":[{"image_path":"https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png"}]}' -H 'Content-Type: application/json')
    if [[ $result == *"generated.mp4"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs image2video
        exit 1
    fi

}

function stop_docker() {
    cd $WORKPATH/comps/image2video/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
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
