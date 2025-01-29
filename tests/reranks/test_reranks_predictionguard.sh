#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/reranking-predictionguard:comps -f comps/reranks/predictionguard/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/reranking-predictionguard built fail"
        exit 1
    else
        echo "opea/reranking-predictionguard built successful"
    fi
}

function start_service() {
    predictionguard_service_port=9000
    unset http_proxy

    docker run -d --name="test-comps-reranking-predictionguard-server" \
    -p ${predictionguard_service_port}:9000 \
    --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
    -e PREDICTIONGUARD_API_KEY=${PREDICTIONGUARD_API_KEY} \
    opea/reranking-predictionguard:comps

    sleep 1m
}

function validate_microservice() {
    predictionguard_service_port=9000
    result=$(http_proxy="" curl http://${ip_address}:${predictionguard_service_port}/v1/reranking\
        -X POST \
        -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"reranked_docs"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-reranking-predictionguard-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-rerank*")
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
