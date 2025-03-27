#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache:-./data}

export TAG='comps'

export TEXT2SQL_PORT=11700
export LLM_ENDPOINT_PORT=11710


export LLM_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
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


    export TGI_LLM_ENDPOINT="http://${ip_address}:${LLM_ENDPOINT_PORT}"
    unset http_proxy

    cd $WORKPATH/comps/text2sql/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log
    check_tgi_connection "${TGI_LLM_ENDPOINT}/health"
}

function validate_microservice() {
    result=$(http_proxy="" curl http://${ip_address}:${TEXT2SQL_PORT}/v1/text2sql\
        -X POST \
        -d '{"input_text": "Find the total number of Albums.","conn_str": {"user": "'${POSTGRES_USER}'","password": "'${POSTGRES_PASSWORD}'","host": "'${ip_address}'", "port": "5442", "database": "'${POSTGRES_DB}'" }}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"output"* ]]; then
        echo $result
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs text2sql-server > ${LOG_PATH}/text2sql.log
        docker logs tgi-server > ${LOG_PATH}/tgi.log
        exit 1
    fi

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
