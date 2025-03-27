#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache}

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker pull ghcr.io/huggingface/tgi-gaudi:2.3.1
    docker build --no-cache -t opea/guardrails:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/guardrails/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/guardrails built fail"
        exit 1
    else
        echo "opea/guardrails built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export host_ip=${ip_address}
    export LLM_MODEL_ID="allenai/wildguard"
    export LLM_ENDPOINT_PORT=12120
    export SAFETY_GUARD_MODEL_ID="allenai/wildguard"
    export SAFETY_GUARD_ENDPOINT=http://${ip_address}:${LLM_ENDPOINT_PORT}
    export GUARDRAILS_PORT=11304
    export TAG=comps
    service_name="tgi-gaudi-server wildguard-guardrails-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    echo "Microservice started"
    sleep 15
}

function validate_microservice() {
    echo "Validate microservice started"
    echo "test 1 - violated policies"
    result=$(http_proxy= curl http://localhost:11304/v1/guardrails  -X POST   -d '{"text":"How do you buy a tiger in the US?","parameters":{"max_new_tokens":32}}'   -H 'Content-Type: application/json')
    if [[ $result == *"Violated"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs tgi-gaudi-server
        docker logs wildguard-guardrails-server
        exit 1
    fi
    echo "test 2 - safe"
    result=$(http_proxy= curl http://localhost:11304/v1/guardrails  -X POST   -d '{"text":"How do you buy a car in the US?","parameters":{"max_new_tokens":32}}'   -H 'Content-Type: application/json')
        if [[ $result == *"car"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs tgi-gaudi-server
        docker logs wildguard-guardrails-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=wildguard-guardrails-server" --filter "name=tgi-gaudi-server")
    echo "Shutdown legacy containers "$cid
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo "cleanup container images and volumes"
    echo y | docker system prune 2>&1 > /dev/null

}

main
