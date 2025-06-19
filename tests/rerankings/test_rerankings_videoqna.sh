#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
service_name="reranking-videoqna"


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
    export TEI_RERANKING_PORT=12006
    export RERANK_PORT=10703
    export TEI_RERANKING_ENDPOINT="http://${host_ip}:${TEI_RERANKING_PORT}"
    export TAG=comps
    export host_ip=${host_ip}

    cd $WORKPATH/comps/rerankings/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 1m
}

function validate_microservice() {
    videoqna_service_port=10703

    result=$(\
    http_proxy="" \
    curl -X 'POST' \
        "http://${ip_address}:$videoqna_service_port/v1/reranking" \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "retrieved_docs": [
            {"doc": [{"text": "this is the retrieved text"}]}
        ],
        "initial_query": "this is the query",
        "top_n": 1,
        "metadata": [
            {"other_key": "value", "video":"top_video_name", "timestamp":"20"},
            {"other_key": "value", "video":"second_video_name", "timestamp":"40"},
            {"other_key": "value", "video":"top_video_name", "timestamp":"20"}
        ]
        }')
    if [[ $result == *"this is the query"* ]]; then
        echo "Result correct for the positive case."
    else
        echo "Result wrong for the positive case. Received was $result"
        exit 1
    fi

    # Add test for negative case
    result=$(\
    http_proxy="" \
    curl -X 'POST' \
        "http://${ip_address}:$videoqna_service_port/v1/reranking" \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "retrieved_docs": [
            {"doc": [{"text": "this is the retrieved text"}]}
        ],
        "initial_query": "this is the query",
        "top_n": 1,
        "metadata": [{"other_key": "value", "video":"top_video_name_bad_format.avi", "timestamp":"20"}]}')
    if [[ $result == *"Invalid file extension"* ]]; then
        echo "Result correct for the negative case."
    else
        echo "Result wrong for the negative case. Received was $result"
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
