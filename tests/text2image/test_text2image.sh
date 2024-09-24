#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/sd:latest --build-arg MODEL=stabilityai/stable-diffusion-xl-base-1.0 -f comps/text2image/dependency/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/sd built fail"
        exit 1
    else
        echo "opea/sd built successful"
    fi
    docker build --no-cache -t opea/text2image:latest -f comps/text2image/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2image built fail"
        exit 1
    else
        echo "opea/text2image built successful"
    fi
}

function start_service() {
    unset http_proxy
    docker run -d --name="test-comps-text2image-sd" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 9378:9378 --ipc=host opea/sd:latest
    docker run -d --name="test-comps-text2image" -e SD_ENDPOINT=http://$ip_address:9378 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 9379:9379 --ipc=host opea/text2image:latest
    sleep 3m
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:9379/v1/text2image -XPOST -d '{"prompt":"An astronaut riding a green horse", "num_images_per_prompt":1}' -H 'Content-Type: application/json')
    if [[ $result == *"images"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs test-comps-text2image-sd
        docker logs test-comps-text2image
        exit 1
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-text2image*")
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
