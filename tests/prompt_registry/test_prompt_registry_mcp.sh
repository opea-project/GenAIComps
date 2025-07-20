#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export DB_NAME=${DB_NAME:-"Prompts"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test_mcp"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/promptregistry-mongo:mcp-test --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/promptregistry-mongo built fail"
        exit 1
    else
        echo "opea/promptregistry-mongo built successful"
    fi
}

function test_mcp_disabled() {
    echo "Testing with MCP disabled (backward compatibility)..."
    cd $WORKPATH
    export PROMPT_REGISTRY_PORT=10601
    export TAG=mcp-test
    export ENABLE_MCP=false
    cd comps/prompt_registry/deployment/docker_compose/

    # Start MongoDB
    docker run -d -p ${MONGO_PORT}:27017 --name=mongo-mcp-disabled mongo:7.0.11
    sleep 5s

    # Start prompt registry with MCP disabled
    docker run -d --name="promptregistry-mongo-server-disabled" \
        -p ${PROMPT_REGISTRY_PORT}:6018 \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e MONGO_HOST=${MONGO_HOST} \
        -e MONGO_PORT=${MONGO_PORT} \
        -e DB_NAME=${DB_NAME} \
        -e COLLECTION_NAME=${COLLECTION_NAME} \
        -e ENABLE_MCP=${ENABLE_MCP} \
        opea/promptregistry-mongo:mcp-test

    sleep 10s

    # Test regular HTTP endpoint
    result=$(curl -X 'POST' \
      http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/create \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "prompt_text": "Test MCP disabled",
        "user": "test_backward_compat"
      }')

    echo "Response: $result"

    # Extract prompt_id from result
    prompt_id=$(echo $result | grep -oP '"prompt_id":"\K[^"]+')

    if [[ ! -z "$prompt_id" ]]; then
        echo "Backward compatibility test PASSED - service works with MCP disabled"

        # Test get endpoint
        get_result=$(curl -X 'POST' \
          http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
          -H 'accept: application/json' \
          -H 'Content-Type: application/json' \
          -d '{
            "user": "test_backward_compat",
            "prompt_id": "'$prompt_id'"
          }')
        echo "Get result: $get_result"

        # Test delete endpoint
        delete_result=$(curl -X 'POST' \
          http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/delete \
          -H 'accept: application/json' \
          -H 'Content-Type: application/json' \
          -d '{
            "user": "test_backward_compat",
            "prompt_id": "'$prompt_id'"
          }')
        echo "Delete result: $delete_result"
    else
        echo "Backward compatibility test FAILED"
        docker logs promptregistry-mongo-server-disabled
        exit 1
    fi

    # Verify SSE endpoint is NOT available when MCP is disabled
    sse_response=$(curl -s -o /dev/null -w "%{http_code}" http://$ip_address:${PROMPT_REGISTRY_PORT}/sse)
    if [[ $sse_response -eq 404 ]]; then
        echo "SSE endpoint correctly returns 404 when MCP is disabled"
    else
        echo "ERROR: SSE endpoint should not be available when MCP is disabled"
        exit 1
    fi

    docker stop promptregistry-mongo-server-disabled mongo-mcp-disabled
    docker rm promptregistry-mongo-server-disabled mongo-mcp-disabled
    sleep 5s
}

function test_mcp_enabled() {
    echo "Testing with MCP enabled..."
    cd $WORKPATH
    export PROMPT_REGISTRY_PORT=10602
    export TAG=mcp-test
    export ENABLE_MCP=true
    cd comps/prompt_registry/deployment/docker_compose/

    # Start MongoDB
    docker run -d -p $((MONGO_PORT+1)):27017 --name=mongo-mcp-enabled mongo:7.0.11
    sleep 5s

    # Start prompt registry with MCP enabled
    docker run -d --name="promptregistry-mongo-server-enabled" \
        -p ${PROMPT_REGISTRY_PORT}:6018 \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e MONGO_HOST=${MONGO_HOST} \
        -e MONGO_PORT=$((MONGO_PORT+1)) \
        -e DB_NAME=${DB_NAME} \
        -e COLLECTION_NAME=${COLLECTION_NAME} \
        -e ENABLE_MCP=${ENABLE_MCP} \
        opea/promptregistry-mongo:mcp-test

    sleep 10s

    # Note: When MCP is enabled in the current implementation,
    # regular HTTP endpoints are not available - only SSE endpoint for MCP
    echo "Note: With MCP enabled, regular HTTP endpoints are not available in current implementation"

    # Test SSE endpoint is available
    sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Accept: text/event-stream" \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/sse)

    if [[ $sse_response -eq 200 ]]; then
        echo "SSE endpoint is available when MCP is enabled (HTTP 200)"
    else
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs promptregistry-mongo-server-enabled
        exit 1
    fi

    # Test MCP initialization
    echo "Testing MCP initialization..."

    # Create a simple MCP test script
    cat > /tmp/test_mcp_init.py << 'EOF'
import asyncio
import json
import sys
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

async def test_mcp():
    try:
        server_url = sys.argv[1]
        async with sse_client(server_url + "/sse") as streams:
            async with ClientSession(*streams) as session:
                result = await session.initialize()
                # Check if we got the expected tool names
                tools = [tool.name for tool in result.tools]
                expected_tools = ["create_prompt", "get_prompt", "delete_prompt"]

                missing_tools = [t for t in expected_tools if t not in tools]
                if missing_tools:
                    print(f"Missing expected tools: {missing_tools}")
                    return False

                print(f"MCP initialization successful. Found tools: {tools}")
                return True
    except Exception as e:
        print(f"MCP test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp())
    sys.exit(0 if success else 1)
EOF

    # Run the MCP test
    python3 /tmp/test_mcp_init.py http://$ip_address:${PROMPT_REGISTRY_PORT}

    if [ $? -ne 0 ]; then
        echo "MCP initialization test failed"
        docker logs promptregistry-mongo-server-enabled
        exit 1
    else
        echo "MCP initialization test passed"
    fi

    rm /tmp/test_mcp_init.py

    docker stop promptregistry-mongo-server-enabled mongo-mcp-enabled
    docker rm promptregistry-mongo-server-enabled mongo-mcp-enabled
    sleep 5s
}

function stop_docker() {
    # Stop any running containers from previous tests
    containers="promptregistry-mongo-server-disabled promptregistry-mongo-server-enabled mongo-mcp-disabled mongo-mcp-enabled"
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

    # Test backward compatibility
    test_mcp_disabled

    # Test MCP functionality
    test_mcp_enabled

    stop_docker
    echo y | docker system prune -f

    echo "All MCP tests completed successfully!"
}

main
