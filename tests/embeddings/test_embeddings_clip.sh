#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy  -f comps/embeddings/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/embedding built fail"
        exit 1
    else
        echo "opea/embedding built successful"
    fi

    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding-multimodal-clip:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/clip/src/Dockerfile .

    if [ $? -ne 0 ]; then
        echo "opea/embedding-multimodal-clip built fail"
        exit 1
    else
        echo "opea/embedding-multimodal-clip built successful"
    fi
}

function start_service() {
    export TAG=comps
    export host_ip=${ip_address}
    export EMBEDDER_PORT=10203
    export MULTIMODAL_CLIP_EMBEDDER_PORT=10204
    export CLIP_EMBEDDING_ENDPOINT=http://${host_ip}:${MULTIMODAL_CLIP_EMBEDDER_PORT}
    service_name="clip-embedding-server"
    cd $WORKPATH
    cd comps/embeddings/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_service() {
    local INPUT_DATA="$1"
    service_port=10203
    result=$(http_proxy="" curl http://${ip_address}:$service_port/v1/embeddings \
        -X POST \
        -d "$INPUT_DATA" \
        -H 'Content-Type: application/json')
    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs clip-embedding-server
        docker logs multimodal-clip-embedding-server
        exit 1
    fi
}

function validate_microservice() {
    ## Test OpenAI API, input single text
    validate_service \
        '{"input":"What is Deep Learning?"}'

    ## Test OpenAI API, input multiple texts with parameters
    validate_service \
        '{"input":["What is Deep Learning?","How are you?"], "dimensions":100}'
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=clip-embedding-server*" --filter "name=multimodal-clip-embedding-server*")
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
