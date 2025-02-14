#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export DB_NAME=${DB_NAME:-"Conversations"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/chathistory-mongo:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/chathistory/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/chathistory-mongo built fail"
        exit 1
    else
        echo "opea/chathistory-mongo built successful"
    fi
}

function start_service() {
    cd $WORKPATH
    export CHATHISTORY_PORT=11000
    export TAG=comps
    cd comps/chathistory/deployment/docker_compose/
    docker compose up -d
    sleep 10s
}

function validate_microservice() {
    result=$(curl -X 'POST' \
  http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": {
    "messages": "test Messages", "user": "test"
  }
}')
    echo $result
    if [[ ${#result} -eq 26 ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-mongo-server
        exit 1
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=chathistory-mongo-*")
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
