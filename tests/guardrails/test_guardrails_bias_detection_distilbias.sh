#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker build --no-cache -t opea/guardrails-bias-detection:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/bias_detection/Dockerfile .
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

function stop_docker() {
    cid=$(docker ps -aq --filter "name=guardrails-bias-detection-server")
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
