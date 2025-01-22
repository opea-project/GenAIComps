#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/gpt-sovits:$TAG -f comps/tts/src/integrations/dependency/gpt-sovits/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/gpt-sovits built fail"
        exit 1
    else
        echo "opea/gpt-sovits built successful"
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
    export TTS_ENDPOINT=http://$ip_address:9880
    export TTS_COMPONENT_NAME=OPEA_GPTSOVITS_TTS

    docker compose -f comps/tts/deployment/docker_compose/compose_gptsovits.yaml up -d
    sleep 15
}

function validate_microservice() {
    http_proxy="" curl localhost:3002/v1/audio/speech -XPOST -d '{"input":"Hello, who are you? 你好。"}' -H 'Content-Type: application/json' --output speech.mp3

    if [[ $(file speech.mp3) == *"RIFF"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs gpt-sovits-service
        docker logs tts-service
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=gpt-sovits-service" --filter "name=tts-service" --format "{{.Names}}" | xargs -r docker stop
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
