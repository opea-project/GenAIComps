#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    dockerfile_name="comps/guardrails/src/bias_detection/$1"
    docker build --no-cache -t opea/guardrails-bias-detection:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f "${dockerfile_name}" .
    if [ $? -ne 0 ]; then
        echo "opea/guardrails-bias-detection built fail"
        exit 1
    else
        echo "opea/guardrails-bias-detection built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export BIAS_DETECTION_PORT=11301
    export TAG=comps
    service_name="guardrails-bias-detection-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
    echo "Microservice started"
}

function validate_microservice() {
    echo "Validate microservice started"
    echo "test 1 - biased"
    result=$(curl localhost:11301/v1/bias -X POST -d '{"text":"John McCain exposed as an unprincipled politician."}' -H 'Content-Type: application/json')
    if [[ $result == *"Violated"* ]]; then
        echo "Result correct."
    else
        docker logs guardrails-bias-detection-server
        exit 1
    fi
    echo "test 2 - non-biased"
    result=$(curl localhost:11301/v1/bias -X POST -d '{"text":"John McCain described as an unprincipled politician."}' -H 'Content-Type: application/json')
    if [[ $result == *"described"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs guardrails-bias-detection-server
        exit 1
    fi
    echo "Validate microservice completed"
}

function stop_service() {
    cd $WORKPATH/comps/guardrails/deployment/docker_compose/
    docker compose down || true
}

function main() {

    build_docker_images "Dockerfile"
    trap stop_service EXIT

    echo "Test normal env ..."
    start_service
    validate_microservice
    stop_service

    echo "Test with openEuler OS ..."
    build_docker_images "Dockerfile.openEuler"
    start_service
    validate_microservice
    stop_service

    docker system prune -f

}

main
