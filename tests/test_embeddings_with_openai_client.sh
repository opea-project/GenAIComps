#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/embedding-tei:comps -f comps/embeddings/langchain/docker/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/embedding-tei built fail"
        exit 1
    else
        echo "opea/embedding-tei built successful"
    fi
}

function start_service() {
    tei_endpoint=5091
    model="BAAI/bge-base-en-v1.5"
    docker run -d --name="test-comps-embedding-tei-endpoint" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p $tei_endpoint:80 -v ./data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 --model-id $model
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${tei_endpoint}"
    tei_service_port=5092
    docker run -d --name="test-comps-embedding-tei-server" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p ${tei_service_port}:6000 --ipc=host -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT  opea/embedding-tei:comps
    sleep 3m
}

function validate_microservice() {
    tei_service_port=5092
    python3 ${WORKPATH}/tests/validate_svc_with_openai.py $ip_address $tei_service_port "embedding"
    if [ $? -ne 0 ]; then
        docker logs test-comps-embedding-tei-endpoint
        docker logs test-comps-embedding-tei-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-embedding-*")
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
