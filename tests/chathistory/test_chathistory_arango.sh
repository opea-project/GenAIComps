#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export OPEA_STORE_NAME="arangodb"
export ARANGODB_HOST="http://${ip_address}:8529"
export ARANGODB_USERNAME=${ARANGODB_USERNAME-"root"}
export ARANGODB_PASSWORD=${ARANGODB_PASSWORD-"test"}
export ARANGODB_ROOT_PASSWORD=${ARANGODB_ROOT_PASSWORD-"test"}
export ARANGODB_DB_NAME=${ARANGODB_DB_NAME-"_system"}
export ARANGODB_COLLECTION_NAME=${ARANGODB_COLLECTION_NAME-"default"}

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
    export CHATHISTORY_PORT=11000
    export TAG=comps
    cd comps/chathistory/deployment/docker_compose/
    docker compose up -d chathistory-arango
    sleep 10s
}

function validate_microservice() {
    # Test create API
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
    id="${result//\"/}"
    if [[ $id =~ ^default/[0-9]+$ ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-arango-server
        exit 1
    fi

    # Test get_by_id API
    result=$(curl -X 'POST' \
  http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "id": "'${id}'"}')
    echo $result
    rs=$(echo $result | jq -r '.messages')
    if [[ $rs == 'test Messages' ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-arango-server
        exit 1
    fi

        # Test get_by_user API
    result=$(curl -X 'POST' \
  http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test"}')
    echo $result
    if [[ $result == '[{"first_query":"test Messages"'* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-arango-server
        exit 1
    fi

    # Test update API
    result=$(curl -X 'POST' \
  http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "data": {
    "messages": "test Messages update", "user": "test"
  },
  "id": "'${id}'"
}')
    echo $result
    if [[ $result == *'true'* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-arango-server
        exit 1
    fi

    # Test get_by_id API
    result=$(curl -X 'POST' \
  http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "id": "'${id}'"}')
    echo $result
    rs=$(echo $result | jq -r '.messages')
    if [[ $rs == 'test Messages update' ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-arango-server
        exit 1
    fi

    # Test delete API
    result=$(curl -X 'POST' \
  http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/delete \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "id": "'${id}'"}')
    echo $result
    if [[ $result == *'true'* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs chathistory-arango-server
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=chathistory-arango-server" --filter "name=arango-vector-db" --format "{{.Names}}" | xargs -r docker stop
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
