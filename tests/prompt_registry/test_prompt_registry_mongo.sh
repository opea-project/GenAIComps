#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export OPEA_STORE_NAME="mongodb"
export DB_NAME=${DB_NAME:-"Prompts"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/promptregistry:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/promptregistry built fail"
        exit 1
    else
        echo "opea/promptregistry built successful"
    fi
}

function start_service() {
    cd $WORKPATH
    export PROMPT_REGISTRY_PORT=10600
    export TAG=comps
    cd comps/prompt_registry/deployment/docker_compose/
    docker compose up -d promptregistry-mongo
    sleep 10s
}

function validate_microservice() {
    # Test create API
    result=$(curl -X 'POST' \
  http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt_text": "test prompt", "user": "test"
}')
    echo $result
    id=""
    if [[ ${#result} -eq 26 ]]; then
        echo "Correct result."
        id="${result//\"/}"
    else
        echo "Incorrect result."
        docker logs promptregistry-mongo-server
        exit 1
    fi

    # Test get_by_id API
    result=$(curl -X 'POST' \
  http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "prompt_id": "'${id}'"}')
    echo $result
    if [[ "${result//\"/}" == "test prompt" ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs promptregistry-mongo-server
        exit 1
    fi

    # Test get_by_user API
    result=$(curl -X 'POST' \
  http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test"}')
    echo $result
    if [[ $result == '[{"prompt_text":"'* ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs promptregistry-mongo-server
        exit 1
    fi

    # Test search API
    result=$(curl -X 'POST' \
  http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "prompt_text": "test prompt"}')
    echo $result
    if [[ $result == '[{"prompt_text":"'* ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs promptregistry-mongo-server
        exit 1
    fi

    # Test delete API
    result=$(curl -X 'POST' \
  http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/delete \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "prompt_id": "'${id}'"}')
    echo $result
    if [[ $result == "true" ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs promptregistry-mongo-server
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=promptregistry-mongo-server" --filter "name=mongodb" --format "{{.Names}}" | xargs -r docker stop
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
