#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export OPEA_STORE_NAME="mongodb"
export DB_NAME=${DB_NAME:-"Feedback"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/feedbackmanagement:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/feedback_management/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/feedbackmanagement built fail"
        exit 1
    else
        echo "opea/feedbackmanagement built successful"
    fi
}

function start_service() {
    cd $WORKPATH
    export FEEDBACK_MANAGEMENT_PORT=11200
    export TAG=comps
    cd comps/feedback_management/deployment/docker_compose/
    docker compose up -d feedbackmanagement-mongo
    sleep 10s
}

function validate_microservice() {
    # Test create API
    result=$(curl -X 'POST' \
  http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/create \
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
    id=""
    if [[ ${#result} -eq 26 ]]; then
        echo "Correct result."
        id="${result//\"/}"
    else
        echo "Incorrect result."
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

    # Test update API
    result=$(curl -X 'POST' \
  http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/create \
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
    "comment": "Fair and Moderate answer",
    "rating": 2,
    "is_thumbs_up": true
  },
  "feedback_id": "'${id}'"
}')
    echo $result
    if [[ $result == "true" ]]; then
        echo "Correct result."
    else
        echo "Incorrect result."
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

    # Test get_by_user API
    result=$(curl -X 'POST' \
  http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test"}')
    echo $result
    if [[ $result == '[{"chat_data":{"messages":'* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

    # Test get_by_id API
    result=$(curl -X 'POST' \
  http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/get \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "feedback_id": "'${id}'"}')
    echo $result
    if [[ $result == '{"chat_data":{"messages":'*'"rating":2'* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

    # Test delete API
    result=$(curl -X 'POST' \
  http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/delete \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"user": "test", "feedback_id": "'${id}'"}')
    echo $result
    if [[ $result == 'true' ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=feedbackmanagement-mongo-server" --filter "name=mongodb" --format "{{.Names}}" | xargs -r docker stop
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
