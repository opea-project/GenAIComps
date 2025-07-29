#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export DB_NAME=${DB_NAME:-"Feedback"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test_mcp"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/feedbackmanagement-mongo:mcp-test --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/feedback_management/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/feedbackmanagement-mongo built fail"
        exit 1
    else
        echo "opea/feedbackmanagement-mongo built successful"
    fi
}

function test_mcp_disabled() {
    echo "Testing with MCP disabled (backward compatibility)..."
    cd $WORKPATH
    export FEEDBACK_MANAGEMENT_PORT=11201
    export TAG=mcp-test
    export ENABLE_MCP=false
    cd comps/feedback_management/deployment/docker_compose/
    docker compose up -d
    sleep 10s

    # Test regular HTTP endpoint
    result=$(curl -X 'POST' \
      http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/create \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "chat_id": "66445d4f71c7eff23d44f78d",
      "chat_data": {
        "user": "test_backward_compat",
        "messages": [
          {
            "role": "user",
            "content": "Test MCP disabled"
          }
        ]
      },
      "feedback_data": {
        "comment": "Testing backward compatibility",
        "rating": 4,
        "is_thumbs_up": true
      }
    }')

    echo "Response: $result"
    if [[ ${#result} -eq 26 ]]; then
        echo "Backward compatibility test PASSED - service works with MCP disabled"
    else
        echo "Backward compatibility test FAILED"
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

    # Verify SSE endpoint is NOT available when MCP is disabled
    sse_response=$(curl -s -o /dev/null -w "%{http_code}" http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/sse)
    if [[ $sse_response -eq 404 ]]; then
        echo "SSE endpoint correctly returns 404 when MCP is disabled"
    else
        echo "ERROR: SSE endpoint should not be available when MCP is disabled"
        exit 1
    fi

    docker compose down
    sleep 5s
}

function test_mcp_enabled() {
    echo "Testing with MCP enabled..."
    cd $WORKPATH
    export FEEDBACK_MANAGEMENT_PORT=11202
    export TAG=mcp-test
    export ENABLE_MCP=true
    cd comps/feedback_management/deployment/docker_compose/
    docker compose up -d
    sleep 10s

    # Test regular HTTP endpoint still works
    result=$(curl -X 'POST' \
      http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/v1/feedback/create \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "chat_id": "66445d4f71c7eff23d44f78e",
      "chat_data": {
        "user": "test_mcp_enabled",
        "messages": [
          {
            "role": "user",
            "content": "Test MCP enabled"
          }
        ]
      },
      "feedback_data": {
        "comment": "Testing MCP functionality",
        "rating": 5,
        "is_thumbs_up": true
      }
    }')

    echo "Response: $result"
    if [[ ${#result} -eq 26 ]]; then
        echo "HTTP endpoint test PASSED with MCP enabled"
    else
        echo "HTTP endpoint test FAILED with MCP enabled"
        docker logs feedbackmanagement-mongo-server
        exit 1
    fi

    # Test SSE endpoint is available
    sse_response=$(curl -s -o /dev/null -w "%{http_code}" http://$ip_address:${FEEDBACK_MANAGEMENT_PORT}/sse)
    if [[ $sse_response -eq 200 ]] || [[ $sse_response -eq 405 ]]; then
        echo "SSE endpoint is available when MCP is enabled"
    else
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        exit 1
    fi

    # MCP functionality is verified through the SSE endpoint availability check above
    echo "MCP functionality verified - SSE endpoint is available"

    docker compose down
    sleep 5s
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=feedbackmanagement-mongo-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    cid=$(docker ps -aq --filter "name=mongodb")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {
    stop_docker

    build_docker_images

    # Test backward compatibility
    test_mcp_disabled

    # Test MCP functionality
    test_mcp_enabled

    stop_docker
    echo y | docker system prune

    echo "All MCP tests completed successfully!"
}

main
