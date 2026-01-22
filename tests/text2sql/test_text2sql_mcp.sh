#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(git rev-parse --show-toplevel)
LOG_PATH="$WORKPATH/tests"
host_addr=${MCP_HOST:-127.0.0.1}
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache:-./data}

export TAG='comps'
export ENABLE_MCP=true
export TEXT2SQL_PORT=11700
export LLM_ENDPOINT_PORT=11710
export host_ip=${host_ip:-$ip_address}

export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN=${HF_TOKEN}
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=testpwd
export POSTGRES_DB=chinook

export service_name="text2sql"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/text2sql:$TAG -f comps/text2sql/src/Dockerfile .
}

check_tgi_connection() {
  url=$1
  timeout=1200
  interval=10

  local start_time=$(date +%s)

  while true; do
    if curl --silent --head --fail "$url" > /dev/null; then
      echo "Success"
      return 0
    fi
    echo
    local current_time=$(date +%s)

    local elapsed_time=$((current_time - start_time))

    if [ "$elapsed_time" -ge "$timeout" ]; then
      echo "Timeout,$((timeout / 60))min can't connect $url"
      return 1
    fi
    echo "Waiting for service for $elapsed_time seconds"
    sleep "$interval"
  done
}

function start_service() {
    export TGI_LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    unset http_proxy

    cd $WORKPATH/comps/text2sql/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log 2>&1
    if [ $? -ne 0 ]; then
        cat ${LOG_PATH}/start_services_with_compose.log
        exit 1
    fi
    check_tgi_connection "${TGI_LLM_ENDPOINT}/health"
}

function validate_microservice() {
    echo "Waiting for SSE endpoint availability..."
    local attempt=0
    local max_attempts=24
    local sse_response=000
    while [[ $attempt -lt $max_attempts ]]; do
        sse_response=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            -H "Accept: text/event-stream" \
            http://$host_addr:${TEXT2SQL_PORT}/sse)
        if [[ $sse_response -eq 200 ]]; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 5
    done

    if [[ $sse_response -ne 200 ]]; then
        echo "ERROR: SSE endpoint should be available when MCP is enabled (got HTTP $sse_response)"
        docker logs text2sql-server
        exit 1
    fi

    pip install mcp
    python3 $WORKPATH/tests/text2sql/validate_mcp.py \
        $host_addr \
        $TEXT2SQL_PORT \
        $POSTGRES_USER \
        $POSTGRES_PASSWORD \
        $host_ip \
        $POSTGRES_DB
}

function stop_docker() {
    cd $WORKPATH/comps/text2sql/deployment/docker_compose
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
