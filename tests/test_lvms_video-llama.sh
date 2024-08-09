#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd) 
    docker build --no-cache -t opea/video-llama-lvm-server:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/video-llama/server/docker/Dockerfile .
    docker build --no-cache -t opea/lvm-video-llama:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy  -f comps/lvms/video-llama/Dockerfile .
    
}

function start_service() {
    cd $WORKPATH
    export no_proxy=$no_proxy,$ip_address
    export LVM_ENDPOINT=http://$ip_address:9009

    docker compose -f comps/lvms/video-llama/docker_compose.yaml up -d

    echo "Waiting for the service to start, downloading model..."
    sleep 1m

    until docker logs video-llama-lvm-server 2>&1 | grep -q "Uvicorn running on"; do
        sleep 5m
    done
}

function validate_microservice() {
    result=$(http_proxy="" curl http://$ip_address:9000/v1/lvm -X POST -d '{"video_url":"https://github.com/DAMO-NLP-SG/Video-LLaMA/raw/main/examples/silence_girl.mp4","chunk_start": 0,"chunk_duration": 9,"prompt":"What is the person doing?","max_new_tokens": 50}' -H 'Content-Type: application/json')
    if [[ $result == *"silence"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=video-llama")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    if docker volume ls | grep -q video-llama-model; then docker volume rm video-llama_video-llama-model; fi

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
