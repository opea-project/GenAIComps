#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export SPEECHT5_PORT=11800
export TTS_PORT=11801

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    dockerfile_name="comps/third_parties/speecht5/src/$1"
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/speecht5:$TAG -f ${dockerfile_name} .
    if [ $? -ne 0 ]; then
        echo "opea/speecht5 built fail"
        exit 1
    else
        echo "opea/speecht5 built successful"
    fi

    dockerfile_name="comps/tts/src/$1"
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/tts:$TAG -f ${dockerfile_name} .
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
    http_proxy="" curl localhost:$TTS_PORT/v1/audio/speech -XPOST -d '{"input":"Hello, who are you?"}' -H 'Content-Type: application/json' --output speech.mp3
    if [[ $(file speech.mp3) == *"RIFF"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs speecht5-service
        docker logs tts-speecht5-service
        exit 1
    fi

}

function stop_service() {
    cd $WORKPATH/comps/tts/deployment/docker_compose/
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
