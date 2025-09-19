#!/bin/bash
# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identifier: Apache-2.0

set -x  # Print commands and their arguments as they are executed

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')  # Adjust to a more reliable command
if [ -z "$ip_address" ]; then
    ip_address="localhost"  # Default to localhost if IP address is empty
fi
export TAG=comps
export PREDICTIONGUARD_PORT=11504
# export LVM_PORT=11505


function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm-pg:comps -f comps/third_parties/predictionguard/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/lvm-pg build failed"
        exit 1
    else
        echo "opea/lvm-pg built successfully"
    fi
}

function start_service() {

    unset http_proxy

    export LVM_ENDPOINT=http://$ip_address:$PREDICTIONGUARD_PORT
    export LVM_COMPONENT_NAME=OPEA_PREDICTION_GUARD_LVM
    docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up predictionguard-service -d

    sleep 60  # Sleep for 1 minute to allow the service to start
}

function validate_microservice() {
    result=$(http_proxy="" curl http://${ip_address}:${PREDICTIONGUARD_PORT}/v1/lvm \
        -X POST \
        -d '{"image": "https://raw.githubusercontent.com/docarray/docarray/main/tests/toydata/image-data/apple.png", "prompt": "Describe the image.", "max_new_tokens": 100}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"text"* ]]; then
        echo "Service response is correct."
    else
        echo "Result wrong. Received was $result"
        docker logs predictionguard-service
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=predictionguard-service" --format "{{.Names}}" | xargs -r docker stop
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
