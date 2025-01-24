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
    docker build --no-cache -t opea/pii-detection-predictionguard:comps -f comps/guardrails/src/pii_detection/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/pii-detection-predictionguard build failed"
        exit 1
    else
        echo "opea/pii-detection-predictionguard built successfully"
    fi
}

function start_service() {
    export PII_PREDICTIONGUARD_PORT=11306
    export TAG=comps
    service_name="pii-predictionguard-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_microservice() {
    pii_service_port=11306
    result=$(http_proxy="" curl http://${ip_address}:${pii_service_port}/v1/pii \
        -X POST \
        -d '{"prompt": "My name is John Doe and my phone number is 123-456-7890.", "replace": true, "replace_method": "mask"}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"new_prompt"* || $result == *"detected_pii"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs pii-predictionguard-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=pii-predictionguard-server")
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
