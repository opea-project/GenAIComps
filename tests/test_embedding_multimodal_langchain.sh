#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding-tei:comps -f comps/embeddings/langchain/docker/Dockerfile .
    docker build --no-cache -t opea/embedding-multimodal:comps -f comps/embeddings/langchain_multimodal/docker/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/embedding-multimodal built fail"
        exit 1
    else
        echo "opea/embedding-multimodal built successful"
    fi
}

function start_service() {
    docker run -d --name="test-embedding-multimodal-server" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 6000:6000 --ipc=host  opea/embedding-multimodal:comps
    sleep 3m
}

function validate_microservice() {
    service_port=6000
    result=$(http_proxy="" curl http://${ip_address}:$service_port/v1/embeddings \
        -X POST \
        -d '{"text":"how many cars are in this image?"}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-embedding-multimodal-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-embedding-multimodal-server-*")
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

