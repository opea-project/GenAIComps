#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker build --no-cache -t opea/guardrails-pii-detection:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/pii_detection/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/guardrails-pii-detection built fail"
        exit 1
    else
        echo "opea/guardrails-pii-detection built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export pii_detection_port=9081
    docker run -d --runtime=runc --name="test-comps-guardrails-pii-detection-endpoint" -p $pii_detection_port:$pii_detection_port --ipc=host -e PII_DETECTION_PORT=$pii_detection_port -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy-$no_proxy opea/guardrails-pii-detection:comps
    sleep 25
    echo "Microservice started"
}

function validate_microservice() {
    echo "Validate microservice started"
    echo "test 1 - PII"
    result=$(curl localhost:9081/v1/pii -X POST -d '{"text":"My name is John Doe and my phone number is (555) 555-5555."}' -H 'Content-Type: application/json')
    if [[ $result == *"PERSON"* && $result == *"PHONE_NUMBER"* ]]; then
        echo "Result correct."
    else
        docker logs test-comps-guardrails-pii-detection-endpoint
        exit 1
    fi
    echo "test 2 - no PII"
    result=$(curl localhost:9081/v1/pii -X POST -d '{"text":"How to write a paper on raising dogs?"}' -H 'Content-Type: application/json')
    if [[ $result == *"paper"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs test-comps-guardrails-pii-detection-endpoint
        exit 1
    fi
    echo "Validate microservice completed"
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-guardrails-pii-detection-endpoint")
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
    echo y | docker system prune > /dev/null 2>&1

}

main
