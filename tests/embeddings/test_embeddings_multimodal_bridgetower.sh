#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export your_mmei_port=12400
export EMBEDDER_PORT=$your_mmei_port
export MMEI_EMBEDDING_ENDPOINT="http://$ip_address:$your_mmei_port"
export your_embedding_port_microservice=10202
export MM_EMBEDDING_PORT_MICROSERVICE=$your_embedding_port_microservice
unset http_proxy

function build_mm_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/embeddings/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/embedding built fail"
        exit 1
    else
        echo "opea/embedding built successfully"
    fi
}

function build_embedding_service_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding-multimodal-bridgetower:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/bridgetower/src/Dockerfile .

    if [ $? -ne 0 ]; then
        echo "opea/embedding-multimodal-bridgetower built fail"
        exit 1
    else
        echo "opea/embedding-multimodal-bridgetower built successful"
    fi
}

function build_docker_images() {
    build_mm_docker_images
    build_embedding_service_images
}

function start_service() {
    service_name="multimodal-bridgetower-embedding-serving multimodal-bridgetower-embedding-server"
    cd $WORKPATH
    cd comps/embeddings/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 30
}

function validate_microservice_text_embedding() {
    result=$(http_proxy="" curl http://${ip_address}:$MM_EMBEDDING_PORT_MICROSERVICE/v1/embeddings \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text" : "This is some sample text."}')

    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs multimodal-bridgetower-embedding-serving
        docker logs multimodal-bridgetower-embedding-server
        exit 1
    fi
}

function validate_microservice_image_text_pair_embedding() {
    result=$(http_proxy="" curl http://${ip_address}:$MM_EMBEDDING_PORT_MICROSERVICE/v1/embeddings \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": {"text" : "This is some sample text."}, "image" : {"url": "https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true"}}')

    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs multimodal-bridgetower-embedding-serving
        docker logs multimodal-bridgetower-embedding-server
        exit 1
    fi
}

function validate_microservice_b64_image_text_pair_embedding() {
    result=$(http_proxy="" curl http://${ip_address}:$MM_EMBEDDING_PORT_MICROSERVICE/v1/embeddings \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"text": {"text" : "This is some sample text."}, "image" : {"base64_image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC"}}')

    if [[ $result == *"embedding"* ]] && [[ $result == *"base64_image"* ]] ; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs multimodal-bridgetower-embedding-serving
        docker logs multimodal-bridgetower-embedding-server
        exit 1
    fi
}

function validate_microservice() {
    validate_microservice_text_embedding
    validate_microservice_image_text_pair_embedding
    validate_microservice_b64_image_text_pair_embedding
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=multimodal-bridgetower-embedding-*")
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
