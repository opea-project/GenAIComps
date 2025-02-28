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
    docker build --no-cache -t opea/toxicity-predictionguard:comps -f comps/guardrails/src/toxicity_detection/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/toxicity-predictionguard build failed"
        exit 1
    else
        echo "opea/toxicity-predictionguard built successfully"
    fi
}

function start_service() {
    export TOXICITY_PREDICTIONGUARD_PORT=11308
    export TAG=comps
    service_name="toxicity-predictionguard-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_microservice() {
    toxicity_service_port=11308
    result=$(http_proxy="" curl http://${ip_address}:${toxicity_service_port}/v1/toxicity \
        -X POST \
        -d '{"text": "I hate you."}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"score"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs toxicity-predictionguard-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=toxicity-predictionguard-server")
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
