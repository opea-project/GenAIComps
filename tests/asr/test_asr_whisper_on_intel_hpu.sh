#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export WHISPER_PORT=10102
export ASR_PORT=10103

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/whisper-gaudi:$TAG --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/whisper/src/Dockerfile.intel_hpu .

    if [ $? -ne 0 ]; then
        echo "opea/whisper-gaudi built fail"
        exit 1
    else
        echo "opea/whisper-gaudi built successful"
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

    docker compose -f comps/asr/deployment/docker_compose/compose.yaml up whisper-gaudi-service asr-whisper-gaudi -d
    sleep 15
}

function validate_microservice() {
    wget https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav
    result=$(http_proxy="" curl http://localhost:$ASR_PORT/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@./sample.wav" -F model="openai/whisper-small")
    rm -f sample.wav
    if [[ $result == *"who is"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs whisper-gaudi-service
        docker logs asr-whisper-gaudi-service
        exit 1
    fi

    wget https://github.com/intel/intel-extension-for-transformers/raw/refs/tags/v1.5/intel_extension_for_transformers/neural_chat/ui/customized/talkingbot/src/lib/components/talkbot/assets/mid-age-man.mp3 -O sample.mp3
    result=$(http_proxy="" curl http://localhost:$ASR_PORT/v1/audio/transcriptions -H "Content-Type: multipart/form-data" -F file="@./sample.mp3" -F model="openai/whisper-small")
    rm -f sample.mp3
    if [[ $result == *"welcome to"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs whisper-gaudi-service
        docker logs asr-whisper-gaudi-service
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=whisper-gaudi-service" --filter "name=asr-whisper-gaudi-service" --format "{{.Names}}" | xargs -r docker stop
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
