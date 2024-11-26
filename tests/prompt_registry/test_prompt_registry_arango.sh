#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export ARANGO_HOST=${ip_address}
export ARANGO_PORT=8529
export ARANGO_PROTOCOL=${ARANGO_PROTOCOL:-"http"}
export ARANGO_USERNAME=${ARANGO_USERNAME:-"root"}
export ARANGO_PASSWORD=${ARANGO_PASSWORD:-"test"}
export DB_NAME=${DB_NAME:-"Prompts"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker run -d -p 8529:8529 --name=test-comps-arango arangodb/arangodb:latest

    docker build --no-cache -t opea/promptregistry-arango-server:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/arango/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/promptregistry-arango-server built fail"
        exit 1
    else
        echo "opea/promptregistry-arango-server built successful"
    fi
}

function start_service() {

    docker run -d --name="test-comps-promptregistry-arango-server" \
        -p 6018:6018 \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e ARANGO_HOST=${ARANGO_HOST} \
        -e ARANGO_PORT=${ARANGO_PORT} \
        -e ARANGO_PROTOCOL=${ARANGO_PROTOCOL} \
        -e ARANGO_USERNAME=${ARANGO_USERNAME} \
        -e ARANGO_PASSWORD=${ARANGO_PASSWORD} \
        -e DB_NAME=${DB_NAME} \
        -e COLLECTION_NAME=${COLLECTION_NAME} \
        opea/promptregistry-arango-server:comps

    sleep 10s
}

function validate_microservice() {
    result=$(curl -X 'POST' \
  http://$ip_address:6018/v1/prompt/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_text": "test prompt", "user": "test"
}')
    echo $result
    if [[ ${#result} -eq 26 ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs test-comps-promptregistry-arango-server
        exit 1
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps*")
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
