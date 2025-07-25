#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(cd "$(dirname "$0")/../.." && pwd)
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export DB_NAME=${DB_NAME:-"Conversations"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}
export ENABLE_MCP=True
export CHATHISTORY_PORT=11000
export TAG=comps

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
    cd comps/chathistory/deployment/docker_compose/
    docker compose up -d
    sleep 10s
}

function validate_microservice() {
    pip install mcp

    # Wait for service to be ready
    service_ready=false
    for i in {1..30}; do
        if curl -s http://${ip_address}:${CHATHISTORY_PORT}/v1/health_check > /dev/null; then
            echo "Service is ready"
            service_ready=true
            break
        fi
        echo "Waiting for service to be ready... ($i/30)"
        sleep 2
    done

    if [ "$service_ready" = false ]; then
        echo "Service failed to start after 60 seconds"
        docker logs chathistory-mongo-server
        exit 1
    fi

    # First check HTTP endpoint is working
    echo "Testing HTTP endpoint..."
    result=$(curl -X 'POST' \
        http://${ip_address}:${CHATHISTORY_PORT}/v1/chathistory/create \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
        "data": {
            "messages": [{"role": "user", "content": "test message"}],
            "user": "test"
        }
    }' 2>/dev/null)

    if [[ ${#result} -lt 10 ]]; then
        echo "HTTP endpoint test failed. Response: $result"
        docker logs chathistory-mongo-server
        exit 1
    else
        echo "HTTP endpoint test passed"
    fi

    # Check if SSE endpoint is accessible
    echo "Checking SSE endpoint..."
    sse_status=$(curl -s -o /dev/null -w "%{http_code}" http://${ip_address}:${CHATHISTORY_PORT}/sse)
    echo "SSE endpoint status: $sse_status"

    if [[ "$sse_status" != "200" && "$sse_status" != "405" ]]; then
        echo "SSE endpoint not available (status: $sse_status)"
        docker logs chathistory-mongo-server
        exit 1
    fi

    python3 ${WORKPATH}/tests/utils/validate_svc_with_mcp.py $ip_address $CHATHISTORY_PORT "chathistory"
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
