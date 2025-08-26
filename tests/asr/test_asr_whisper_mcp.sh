#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export WHISPER_PORT=10104
export ASR_PORT=10105
export ENABLE_MCP=True
cd $WORKPATH


function build_docker_images() {
    echo $(pwd)
    docker build --no-cache -t opea/whisper:$TAG --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/whisper/src/Dockerfile .

    if [ $? -ne 0 ]; then
        echo "opea/whisper built fail"
        exit 1
    else
        echo "opea/whisper built successful"
    fi

    docker build --no-cache -t opea/asr:$TAG --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/asr/src/Dockerfile .

    if [ $? -ne 0 ]; then
        echo "opea/asr built fail"
        exit 1
    else
        echo "opea/asr built successful"
    fi
}

function start_service() {
    unset http_proxy
    export ASR_ENDPOINT=http://$ip_address:$WHISPER_PORT

    docker compose -f comps/asr/deployment/docker_compose/compose.yaml up whisper-service asr -d
    sleep 1m
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/asr/validate_svc_with_mcp.py $ip_address $ASR_PORT
    if [ $? -ne 0 ]; then
        docker logs whisper-service
        docker logs asr-service
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=whisper-service" --filter "name=asr-service" --format "{{.Names}}" | xargs -r docker stop
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
