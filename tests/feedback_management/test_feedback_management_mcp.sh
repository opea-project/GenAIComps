#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export OPEA_STORE_NAME="mongodb"
export DB_NAME=${DB_NAME:-"Feedback"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test_mcp"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/feedbackmanagement:mcp-test --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/feedback_management/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/feedbackmanagement built fail"
        exit 1
    else
        echo "opea/feedbackmanagement built successful"
    fi
}


function start_service() {
    echo "Starting feedback management service with MCP enabled..."
    cd $WORKPATH
    export FEEDBACK_MANAGEMENT_PORT=11202
    export TAG=mcp-test
    export ENABLE_MCP=true
    cd comps/feedback_management/deployment/docker_compose/
    docker compose up -d feedbackmanagement-mongo
    sleep 10s
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/feedback_management/validate_svc_with_mcp.py $ip_address $FEEDBACK_MANAGEMENT_PORT
    if [ $? -ne 0 ]; then
        docker logs feedbackmanagement-mongo-server
        docker logs mongodb
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
