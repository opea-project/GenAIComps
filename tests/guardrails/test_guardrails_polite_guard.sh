#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker build --no-cache -t opea/guardrails-polite-guard:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/polite_guard/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/guardrails-polite-guard built fail"
        exit 1
    else
        echo "opea/guardrails-polite-guard built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export POLITE_GUARD_PORT=11301
    export TAG=comps
    service_name="guardrails-polite-guard-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
    max_retries=3
    retries=0
    until docker logs ${service_name} 2>&1 | grep -q "Application startup complete"; do
        if [ $retries -ge $max_retries ]; then
            echo "Application failed to start after $max_retries attempts."
            exit 1
        fi
        echo "Waiting for application startup to complete... (Attempt $((retries + 1))/$max_retries)"
        retries=$((retries + 1))
        sleep 2  # Wait for 2 seconds before checking again
    done
    echo "Microservice started"
}

function validate_microservice() {
    echo "Validate microservice started"
    echo "test 1 - Impolite"
    result=$(curl localhost:11301/v1/polite -X POST -d '{"text":"He is stupid"}' -H 'Content-Type: application/json')
    if [[ $result == *"Violated"* ]]; then
        echo "Result correct."
    else
        docker logs guardrails-polite-guard-server
        exit 1
    fi
    echo "test 2 - Polite"
    result=$(curl localhost:11301/v1/polite -X POST -d '{"text":"He is kind"}' -H 'Content-Type: application/json')
    if [[ $result == *"kind"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs guardrails-polite-guard-server
        exit 1
    fi
    echo "Validate microservice completed"
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=guardrails-polite-guard-server")
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
