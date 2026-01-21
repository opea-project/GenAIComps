#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(git rev-parse --show-toplevel)
host_addr=${MCP_HOST:-127.0.0.1}
service_name="text2image"
export ENABLE_MCP=true
export TEXT2IMAGE_PORT=${TEXT2IMAGE_PORT:-9379}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2image:latest -f comps/text2image/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2image built fail"
        exit 1
    else
        echo "opea/text2image built successful"
    fi
}

function start_service() {
    unset http_proxy
    export MODEL=stabilityai/stable-diffusion-xl-base-1.0
    cd $WORKPATH/comps/text2image/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 30s
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    local attempt=0
    local max_attempts=24
    local sse_response=000
    while [[ $attempt -lt $max_attempts ]]; do
        sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            -H "Accept: text/event-stream" \
            http://$host_addr:${TEXT2IMAGE_PORT}/sse)
        if [[ $sse_response -eq 200 ]]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 10s
    done

    if [[ $sse_response -ne 200 ]]; then
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs text2image
        exit 1
    fi

    pip install mcp
    python3 $WORKPATH/tests/text2image/validate_mcp.py $host_addr $TEXT2IMAGE_PORT
    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    cd $WORKPATH/comps/text2image/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
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
