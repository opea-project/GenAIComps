#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/reranking-tei:comps -f comps/reranks/langchain/docker/Dockerfile .
}

function start_service() {
    local_port=5003
    export TEI_RERANKING_ENDPOINT="http://${ip_address}:${local_port}"
    export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
    docker run -d --name="test-comps-reranking-tei-server" -p ${local_port}:${local_port} --ipc=host -e TEI_RERANKING_ENDPOINT=$TEI_RERANKING_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN opea/reranking-tei:comps
}

function validate_microservice() {
    local_port=5003
    curl http://${ip_address}:${local_port}/v1/reranking\
        -X POST \
        -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
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
