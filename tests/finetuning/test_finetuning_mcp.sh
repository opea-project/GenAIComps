#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKPATH=$(cd "${SCRIPT_DIR}/../.." && pwd)
ip_address=127.0.0.1

export FINETUNING_PORT=18015
export TAG=mcp-test
export ENABLE_MCP=true

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/finetuning:$TAG \
        --build-arg https_proxy=$https_proxy \
        --build-arg http_proxy=$http_proxy \
        --build-arg HF_TOKEN=$HF_TOKEN \
        -f comps/finetuning/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/finetuning built fail"
        exit 1
    else
        echo "opea/finetuning built successful"
    fi
}

function start_service() {
    cd $WORKPATH

    docker run -d --name="finetuning-mcp-server" \
        -p ${FINETUNING_PORT}:8015 \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        -e HF_TOKEN=$HF_TOKEN \
        -e ENABLE_MCP=${ENABLE_MCP} \
        --ipc=host \
        opea/finetuning:$TAG

    sleep 10s
}

function validate_microservice() {
    # Wait for SSE endpoint
    echo "Waiting for SSE endpoint availability..."
    sse_response=000
    for i in $(seq 1 24); do
        sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            -H "Accept: text/event-stream" \
            http://$ip_address:${FINETUNING_PORT}/sse)
        if [[ $sse_response -eq 200 ]]; then
            echo "SSE endpoint is available when MCP is enabled (HTTP 200)"
            break
        fi
        echo "SSE not ready (HTTP $sse_response), retrying..."
        sleep 5
    done

    if [[ $sse_response -ne 200 ]]; then
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs finetuning-mcp-server
        exit 1
    fi

    # Install mcp package for testing (avoid system Python)
    echo "Installing mcp package in venv..."
    python3 -m venv .venv-mcp-test
    . .venv-mcp-test/bin/activate
    pip install -q --upgrade pip
    pip install -q mcp

    # Test MCP functionality using the dedicated validation script
    echo "Testing MCP functionality..."
    python3 ${WORKPATH}/tests/finetuning/validate_mcp.py $ip_address ${FINETUNING_PORT}

    deactivate
    rm -rf .venv-mcp-test

    if [ $? -ne 0 ]; then
        echo "MCP validation test failed"
        docker logs finetuning-mcp-server
        exit 1
    else
        echo "MCP validation test passed"
    fi
}

function stop_docker() {
    if docker ps -a | grep -q finetuning-mcp-server; then
        docker stop finetuning-mcp-server 2>/dev/null || true
        docker rm finetuning-mcp-server 2>/dev/null || true
    fi
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
