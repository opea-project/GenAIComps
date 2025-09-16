#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(cd "$(dirname "$0")/../.." && pwd)
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export OPEA_STORE_NAME="mongodb"
export DB_NAME=${DB_NAME:-"Conversations"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}
export ENABLE_MCP=True
export CHATHISTORY_PORT=11000
export TAG=comps

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/chathistory:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/chathistory/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/chathistory built fail"
        exit 1
    else
        echo "opea/chathistory built successful"
    fi
}

function start_service() {
    cd $WORKPATH
    export ENABLE_MCP=True
    cd comps/chathistory/deployment/docker_compose/
    docker compose up -d chathistory-mongo
    sleep 10s
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/chathistory/validate_mcp.py $ip_address $CHATHISTORY_PORT
    if [ $? -ne 0 ]; then
        docker logs mongodb
        docker logs chathistory-mongo-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=chathistory-mongo-*" --filter "name=mongodb")
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
