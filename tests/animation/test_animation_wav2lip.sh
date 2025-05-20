#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export TAG=comps
export ANIMATION_PORT=10900
export WAV2LIP_PORT=12300
export service_name="animation"


function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build -t opea/wav2lip:$TAG -f comps/third_parties/wav2lip/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/wav2lip built fail"
        exit 1
    else
        echo "opea/wav2lip built successful"
    fi
    docker build --no-cache -t opea/animation:$TAG -f comps/animation/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/animation built fail"
        exit 1
    else
        echo "opea/animation built successful"
    fi
}

function start_service() {
    unset http_proxy
    # Set env vars
    export ip_address=$(hostname -I | awk '{print $1}')
    export DEVICE="cpu"
    export INFERENCE_MODE='wav2lip+gfpgan'
    export CHECKPOINT_PATH='/usr/local/lib/python3.11/site-packages/Wav2Lip/checkpoints/wav2lip_gan.pth'
    export FACE="/home/user/comps/animation/src/assets/img/avatar1.jpg"
    export AUDIO='None'
    export FACESIZE=96
    export OUTFILE="/home/user/comps/animation/src/assets/outputs/result.mp4"
    export GFPGAN_MODEL_VERSION=1.4 # latest version, can roll back to v1.3 if needed
    export UPSCALE_FACTOR=1
    export FPS=10
    export WAV2LIP_ENDPOINT="http://$ip_address:$WAV2LIP_PORT"

    cd $WORKPATH/comps/animation/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d

    sleep 3m
}

function validate_microservice() {
    cd $WORKPATH
    result=$(http_proxy="" curl http://localhost:$ANIMATION_PORT/v1/animation -X POST -H "Content-Type: application/json" -d @comps/animation/src/assets/audio/sample_question.json)
    if [[ $result == *"result.mp4"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs wav2lip-server
        docker logs animation-server
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/animation/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
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
