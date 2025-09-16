#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27018
export OPEA_STORE_NAME="mongodb"
export DB_NAME=${DB_NAME:-"Prompts"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test_mcp"}
export PROMPT_REGISTRY_PORT=10602
export TAG=mcp-test
export ENABLE_MCP=true

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/promptregistry:$TAG --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/promptregistry built fail"
        exit 1
    else
        echo "opea/promptregistry built successful"
    fi
}

function start_service() {
    cd $WORKPATH
    cd comps/prompt_registry/deployment/docker_compose/

    # Start MongoDB
    docker run -d -p ${MONGO_PORT}:27017 --name=mongo-mcp mongo:7.0.11
    sleep 5s

    # Start prompt registry with MCP enabled
    docker run -d --name="promptregistry-mongo-server" \
        -p ${PROMPT_REGISTRY_PORT}:6018 \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e MONGO_HOST=${MONGO_HOST} \
        -e MONGO_PORT=${MONGO_PORT} \
        -e OPEA_STORE_NAME=${OPEA_STORE_NAME} \
        -e DB_NAME=${DB_NAME} \
        -e COLLECTION_NAME=${COLLECTION_NAME} \
        -e ENABLE_MCP=${ENABLE_MCP} \
        opea/promptregistry:$TAG

    sleep 10s
}

function validate_microservice() {
    # Test SSE endpoint is available
    echo "Testing SSE endpoint availability..."
    sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 5 \
        -H "Accept: text/event-stream" \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/sse)

    if [[ $sse_response -eq 200 ]]; then
        echo "SSE endpoint is available when MCP is enabled (HTTP 200)"
    else
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs promptregistry-mongo-server
        exit 1
    fi

    # Install mcp package for testing
    echo "Installing mcp package..."
    pip install mcp

    # Test MCP functionality using the dedicated validation script
    echo "Testing MCP functionality..."
    python3 ${WORKPATH}/tests/prompt_registry/validate_mcp.py $ip_address ${PROMPT_REGISTRY_PORT}

    if [ $? -ne 0 ]; then
        echo "MCP validation test failed"
        docker logs promptregistry-mongo-server
        exit 1
    else
        echo "MCP validation test passed"
    fi
}

function stop_docker() {
    # Stop any running containers from previous tests
    containers="promptregistry-mongo-server mongo-mcp"
    for container in $containers; do
        if docker ps -a | grep -q $container; then
            docker stop $container 2>/dev/null || true
            docker rm $container 2>/dev/null || true
        fi
    done
}

function main() {
    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune -f

    echo "MCP test completed successfully!"
}

main
