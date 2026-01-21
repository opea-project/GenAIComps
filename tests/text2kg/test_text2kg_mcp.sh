#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x
WORKPATH=$(git rev-parse --show-toplevel)
LOG_PATH="$WORKPATH/comps/text2kg/deployment/docker_compose"
export DATA_PATH=${model_cache}
source $WORKPATH/comps/text2kg/src/environment_setup.sh

host_addr=${MCP_HOST:-127.0.0.1}
service_name="text2kg"
export ENABLE_MCP=true
export TEXT2KG_PORT=${TEXT2KG_PORT:-8090}

function build_docker() {
    echo "===================  START BUILD DOCKER ========================"
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2kg:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2kg/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2kg built fail"
        exit 1
    else
        echo "opea/text2kg built successful"
    fi
    echo "===================  END BUILD DOCKER ========================"
}

function start_service() {
    echo "===================  START SERVICE ========================"
    cd $WORKPATH/comps/text2kg/deployment/docker_compose
    docker compose -f compose.yaml -f custom-override.yml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log 2>&1
    if [ $? -ne 0 ]; then
        echo "docker compose up failed. Log output:"
        cat ${LOG_PATH}/start_services_with_compose.log
        docker compose -f compose.yaml -f custom-override.yml ps -a || true
        exit 1
    fi

    if ! docker ps -a --format "{{.Names}}" | grep -q "^${service_name}$"; then
        echo "Container ${service_name} was not created."
        cat ${LOG_PATH}/start_services_with_compose.log
        docker compose -f compose.yaml -f custom-override.yml ps -a || true
        exit 1
    fi

    sleep 10s
    echo "===================  END SERVICE ========================"
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    local attempt=0
    local max_attempts=18
    local sse_response=000
    while [[ $attempt -lt $max_attempts ]]; do
        sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            -H "Accept: text/event-stream" \
            http://$host_addr:${TEXT2KG_PORT}/sse)
        if [[ $sse_response -eq 200 ]]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 10s
    done

    if [[ $sse_response -ne 200 ]]; then
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs text2kg
        exit 1
    fi

    pip install mcp
    python3 $WORKPATH/tests/text2kg/validate_mcp.py $host_addr $TEXT2KG_PORT
    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    echo "===================  START STOP DOCKER ========================"
    cd $WORKPATH/comps/text2kg/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
    echo "===================  END STOP DOCKER ========================"
}

function main() {

    stop_docker

    build_docker
    start_service
    validate_microservice

    stop_docker

}

main
