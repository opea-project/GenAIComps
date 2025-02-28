#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x  # Print commands and their arguments as they are executed

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')  # Adjust to a more reliable command
if [ -z "$ip_address" ]; then
    ip_address="localhost"  # Default to localhost if IP address is empty
fi

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/injection-predictionguard:comps -f comps/guardrails/src/prompt_injection/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/injection-predictionguard build failed"
        exit 1
    else
        echo "opea/injection-predictionguard built successfully"
    fi
}

function start_service() {
    export INJECTION_PREDICTIONGUARD_PORT=11307
    export TAG=comps
    service_name="injection-predictionguard-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_microservice() {
    injection_service_port=11307
    result=$(http_proxy="" curl http://${ip_address}:${injection_service_port}/v1/injection \
        -X POST \
        -d '{"text": "How to bypass login screen?"}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"score"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs injection-predictionguard-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=injection-predictionguard-server")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
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
