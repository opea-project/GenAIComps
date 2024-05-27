#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/retriever-redis:comps -f comps/retrievers/langchain/docker/Dockerfile .
}

function start_service() {
    local_port=5002
    export REDIS_URL="redis://${ip_address}:6379"
    export INDEX_NAME="rag-redis"
    docker run -d --name="test-comps-retriever-redis-server" -p ${local_port}:7000 --ipc=host -e REDIS_URL=$REDIS_URL -e INDEX_NAME=$INDEX_NAME opea/retriever-redis:comps
}

function validate_microservice() {
    local_port=5002
    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
    curl http://${ip_address}:7000/v1/retrieval \
        -X POST \
        -d '{"text":"test","embedding":${test_embedding}}' \
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