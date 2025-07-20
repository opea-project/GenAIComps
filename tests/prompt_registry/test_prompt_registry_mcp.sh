#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export MONGO_HOST=${ip_address}
export MONGO_PORT=27017
export DB_NAME=${DB_NAME:-"Prompts"}
export COLLECTION_NAME=${COLLECTION_NAME:-"test"}
export ENABLE_MCP=true

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/promptregistry-mongo-mcp:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/promptregistry-mongo-mcp built fail"
        exit 1
    else
        echo "opea/promptregistry-mongo-mcp built successful"
    fi
}

function start_service() {
    cd $WORKPATH
    export PROMPT_REGISTRY_PORT=10601
    export TAG=comps
    export REGISTRY=opea
    # Override image name for MCP test
    export COMPOSE_PROJECT_NAME=prompt-registry-mcp
    cd comps/prompt_registry/deployment/docker_compose/

    # Start MongoDB
    docker run -d -p ${MONGO_PORT}:27017 --name=mongo-mcp mongo:7.0.11
    sleep 5s

    # Start prompt registry with MCP enabled
    docker run -d --name="promptregistry-mongo-mcp-server" \
        -p ${PROMPT_REGISTRY_PORT}:6018 \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e MONGO_HOST=${MONGO_HOST} \
        -e MONGO_PORT=${MONGO_PORT} \
        -e DB_NAME=${DB_NAME} \
        -e COLLECTION_NAME=${COLLECTION_NAME} \
        -e ENABLE_MCP=${ENABLE_MCP} \
        opea/promptregistry-mongo-mcp:comps

    sleep 10s
}

function validate_mcp_endpoints() {
    # Test regular REST endpoints still work
    echo "Testing REST endpoints with MCP enabled..."

    # Create a prompt
    create_result=$(curl -X 'POST' \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/create \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "prompt_text": "test prompt for MCP",
            "user": "test_mcp"
        }')
    echo "Create result: $create_result"

    # Extract prompt_id from result
    prompt_id=$(echo $create_result | grep -oP '"prompt_id":"\K[^"]+')

    if [[ -z "$prompt_id" ]]; then
        echo "Failed to create prompt"
        docker logs promptregistry-mongo-mcp-server
        exit 1
    fi

    # Get prompt by user
    get_user_result=$(curl -X 'POST' \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "user": "test_mcp"
        }')
    echo "Get by user result: $get_user_result"

    # Get prompt by ID
    get_id_result=$(curl -X 'POST' \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "user": "test_mcp",
            "prompt_id": "'$prompt_id'"
        }')
    echo "Get by ID result: $get_id_result"

    # Search prompt by keyword
    search_result=$(curl -X 'POST' \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "user": "test_mcp",
            "prompt_text": "MCP"
        }')
    echo "Search result: $search_result"

    # Delete prompt
    delete_result=$(curl -X 'POST' \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/delete \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "user": "test_mcp",
            "prompt_id": "'$prompt_id'"
        }')
    echo "Delete result: $delete_result"

    # Verify deletion
    verify_result=$(curl -X 'POST' \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/v1/prompt/get \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "user": "test_mcp",
            "prompt_id": "'$prompt_id'"
        }')
    echo "Verify deletion result: $verify_result"

    if [[ "$verify_result" != "null" ]]; then
        echo "Failed to delete prompt"
        exit 1
    fi
}

function validate_mcp_sse_endpoint() {
    echo "Testing MCP SSE endpoint availability..."

    # Test SSE endpoint exists
    sse_test=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Accept: text/event-stream" \
        http://$ip_address:${PROMPT_REGISTRY_PORT}/sse)

    if [[ "$sse_test" == "200" ]]; then
        echo "MCP SSE endpoint is available (HTTP 200)"
    else
        echo "MCP SSE endpoint not available (HTTP $sse_test)"
        docker logs promptregistry-mongo-mcp-server
        exit 1
    fi

    # Test MCP initialization request
    # This sends a minimal MCP initialization request to the SSE endpoint
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
                # Check if we got the expected service names
                tools = [tool.name for tool in result.tools]
                expected_tools = ["opea_service@prompt_create", "opea_service@prompt_get", "opea_service@prompt_delete"]

                for expected in expected_tools:
                    if expected not in tools:
                        print(f"Missing expected tool: {expected}")
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
        docker logs promptregistry-mongo-mcp-server
        exit 1
    else
        echo "MCP initialization test passed"
    fi

    rm /tmp/test_mcp_init.py
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=promptregistry-mongo-mcp-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi

    cid=$(docker ps -aq --filter "name=mongo-mcp")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {
    stop_docker

    build_docker_images
    start_service

    validate_mcp_endpoints
    validate_mcp_sse_endpoint

    stop_docker
    echo y | docker system prune

    echo "MCP test completed successfully!"
}

main
