#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(git rev-parse --show-toplevel)
LOG_PATH="$WORKPATH/comps/text2graph/deployment/docker_compose"
source $WORKPATH/comps/text2graph/src/setup_service_env.sh

host_addr=${MCP_HOST:-127.0.0.1}
service_name="text2graph"
export ENABLE_MCP=true
export TEXT2GRAPH_PORT=${TEXT2GRAPH_PORT:-8090}

function build_docker_graph() {
    echo "===================  START BUILD DOCKER ========================"
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2graph:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2graph/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2graph built fail"
        exit 1
    else
        echo "opea/text2graph built successful"
    fi
    echo "===================  END BUILD DOCKER ========================"
}

function start_service() {
    echo "===================  START SERVICE ========================"
    cd $WORKPATH/comps/text2graph/deployment/docker_compose
    docker compose up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 10s
    echo "===================  END SERVICE ========================"
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    # Test SSE endpoint is available (wait for service to come up)
    local attempt=0
    local max_attempts=12
    local sse_response=000
    while [[ $attempt -lt $max_attempts ]]; do
        sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            -H "Accept: text/event-stream" \
            http://$host_addr:${TEXT2GRAPH_PORT}/sse)
        if [[ $sse_response -eq 200 ]]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 5s
    done

    if [[ $sse_response -ne 200 ]]; then
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs text2graph
        exit 1
    fi

    pip install mcp
    python3 $WORKPATH/tests/text2graph/validate_mcp.py $host_addr $TEXT2GRAPH_PORT
    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    echo "===================  START STOP DOCKER ========================"
    cd $WORKPATH/comps/text2graph/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
    echo "===================  END STOP DOCKER ========================"
}

function main() {
    stop_docker

    build_docker_graph
    start_service
    validate_microservice

    stop_docker
    echo y | docker system prune
}

main
