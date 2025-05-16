#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export SPEECHT5_PORT=11806
export TTS_PORT=11807
export ENABLE_MCP=True


function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/speecht5:$TAG -f comps/third_parties/speecht5/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/speecht5 built fail"
        exit 1
    else
        echo "opea/speecht5 built successful"
    fi
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/tts:$TAG -f comps/tts/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/tts built fail"
        exit 1
    else
        echo "opea/tts built successful"
    fi
}

function start_service() {
    unset http_proxy
    export TTS_ENDPOINT=http://$ip_address:$SPEECHT5_PORT
    export TTS_COMPONENT_NAME=OPEA_SPEECHT5_TTS

    docker compose -f comps/tts/deployment/docker_compose/compose.yaml up speecht5-service tts-speecht5 -d
    sleep 15
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/utils/validate_svc_with_mcp.py $ip_address $TTS_PORT "tts"
    if [ $? -ne 0 ]; then
        docker logs speecht5-service
        docker logs tts-speecht5-service
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=speecht5-service" --filter "name=tts-speecht5-service" --format "{{.Names}}" | xargs -r docker stop
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
