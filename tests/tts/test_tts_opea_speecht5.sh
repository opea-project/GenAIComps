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
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/speecht5:$TAG -f comps/tts/src/integrations/dependency/speecht5/Dockerfile .
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
    export TTS_ENDPOINT=http://$ip_address:7055

    docker compose -f comps/tts/deployment/docker_compose/compose_speecht5.yaml up -d
    sleep 15
}

function validate_microservice() {
    http_proxy="" curl localhost:3002/v1/audio/speech -XPOST -d '{"input":"Hello, who are you?"}' -H 'Content-Type: application/json' --output speech.mp3
    if [[ $(file speech.mp3) == *"RIFF"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs speecht5-service
        docker logs tts-service
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=speecht5-service" --filter "name=tts-service" --format "{{.Names}}" | xargs -r docker stop
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
