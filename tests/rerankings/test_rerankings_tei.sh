#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
service_name="reranking-tei"
export DATA_PATH=${model_cache}

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache \
          -t opea/reranking:comps \
          --build-arg https_proxy=$https_proxy \
          --build-arg http_proxy=$http_proxy \
          -f comps/rerankings/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/reranking built fail"
        exit 1
    else
        echo "opea/reranking built successful"
    fi
}

function start_service() {
    export RERANK_MODEL_ID="BAAI/bge-reranker-base"
    export TEI_RERANKING_PORT=12003
    export RERANK_PORT=10700
    export TEI_RERANKING_ENDPOINT="http://${host_ip}:${TEI_RERANKING_PORT}"
    export TAG=comps
    export host_ip=${host_ip}

    cd $WORKPATH/comps/rerankings/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 1m
}

function validate_microservice() {
    tei_service_port=10700
    local CONTENT=$(curl http://${host_ip}:${tei_service_port}/v1/reranking \
        -X POST \
        -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
        -H 'Content-Type: application/json')

    if echo "$CONTENT" | grep -q "documents"; then
        echo "Content is as expected."
    else
        echo "Content does not match the expected result: $CONTENT"
        docker logs test-comps-reranking-server
        docker logs test-comps-reranking-endpoint
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/rerankings/deployment/docker_compose
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
