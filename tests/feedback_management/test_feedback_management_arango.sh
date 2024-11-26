#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export ARANGO_HOST=${ip_address}
export ARANGO_PORT=8529
export ARANGO_PROTOCOL=${ARANGO_PROTOCOL:-"http"}
export ARANGO_USERNAME=${ARANGO_USERNAME:-"root"}
export ARANGO_PASSWORD=${ARANGO_PASSWORD:-"test"}
export DB_NAME=${DB_NAME:-"Feedback"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker run -d -p 8529:8529 --name=test-comps-arango arangodb/arangodb:latest

    docker build --no-cache -t opea/feedbackmanagement-arango-server:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/feedback_management/arango/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/feedbackmanagement-arango-server built fail"
        exit 1
    else
        echo "opea/feedbackmanagement-arango-server built successful"
    fi
}

function start_service() {

    docker run -d --name="test-comps-feedbackmanagement-arango-server" \
        -p 6016:6016 \
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
        opea/feedbackmanagement-arango-server:comps

    sleep 10s
}

function validate_microservice() {
    result=$(curl -X 'POST' \
  http://$ip_address:6016/v1/feedback/create \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "chat_id": "66445d4f71c7eff23d44f78d",
  "chat_data": {
    "user": "test",
    "messages": [
      {
        "role": "system",
        "content": "You are helpful assistant"
      },
      {
        "role": "user",
        "content": "hi",
        "time": "1724915247"
      },
      {
        "role": "assistant",
        "content": "Hi, may I help you?",
        "time": "1724915249"
      }
    ]
  },
  "feedback_data": {
    "comment": "Moderate",
    "rating": 3,
    "is_thumbs_up": true
  }
}')
    echo $result
    if [[ ${#result} -eq 26 ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs test-comps-feedbackmanagement-arango-server
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
