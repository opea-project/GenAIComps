#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export VIDEO_LLAMA_PORT=11506
export LVM_PORT=11507

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm-video-llama:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/src/integrations/dependency/video-llama/Dockerfile .
    if $? ; then
        echo "opea/lvm-video-llama built fail"
        exit 1
    else
        echo "opea/lvm-video-llama built successful"
    fi
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy  -f comps/lvms/src/Dockerfile .
    if $? ; then
        echo "opea/lvm built fail"
        exit 1
    else
        echo "opea/lvm built successful"
    fi

}

function start_service() {
    cd $WORKPATH
    unset http_proxy

    export LVM_ENDPOINT=http://$ip_address:$VIDEO_LLAMA_PORT
    export LVM_COMPONENT_NAME=OPEA_VIDEO_LLAMA_LVM

    docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up video-llama-service lvm-video-llama -d

    sleep 15
}

function validate_microservice() {

    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -X POST -d '{"video_url":"silence_girl.mp4","chunk_start": 0,"chunk_duration": 7,"prompt":"What is the person doing?","max_new_tokens": 50}' -H 'Content-Type: application/json')

    if [[ $result == *"silence"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs video-llama-service &> ${LOG_PATH}/video-llama-service.log
        docker logs lvm-video-llama-service &> ${LOG_PATH}/lvm.log
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=video-llama-service" --filter "name=lvm-video-llama-service" --format "{{.Names}}" | xargs -r docker stop

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
