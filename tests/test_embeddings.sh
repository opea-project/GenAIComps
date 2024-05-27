#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/embedding-tei:comps -f comps/embeddings/langchain/docker/Dockerfile .
}

function start_service() {
    local_port=5001
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:$local_port"
    docker run -d --name="test-comps-embedding-tei-server" -p ${local_port}:6000 --ipc=host -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT  opea/embedding-tei:comps

}

function validate_microservice() {
    local_port=5001
    curl localhost:$local_port/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-*")
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