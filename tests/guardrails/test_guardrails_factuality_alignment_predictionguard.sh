#!/bin/bash
# Copyright (C) 2024 Prediction Guard, Inc.
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
    docker build --no-cache -t opea/guardrails-factuality-predictionguard:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/factuality_alignment/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/guardrails-factuality-predictionguard build failed"
        exit 1
    else
        echo "opea/guardrails-factuality-predictionguard built successfully"
    fi
}

function start_service() {
    export FACTUALITY_ALIGNMENT_PORT=11302
    service_name="guardrails-factuality-predictionguard-server"
    export TAG=comps
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_microservice() {
    factuality_service_port=11302
    result=$(http_proxy="" curl http://${ip_address}:${factuality_service_port}/v1/factuality \
        -X POST \
        -d '{"reference": "The Eiffel Tower is in Paris.", "text": "The Eiffel Tower is in Berlin."}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"score"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs guardrails-factuality-predictionguard-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=guardrails-factuality-predictionguard-*")
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
