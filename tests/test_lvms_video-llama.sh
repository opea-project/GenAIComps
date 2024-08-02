#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build -t opea/lvm-video-llama:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/lvms/video-llama/server/docker/Dockerfile .
    # docker build --no-cache -t opea/lvm:latest -f comps/lvms/Dockerfile .
    
}

function start_service() {
    cd $WORKPATH
    no_proxy=$no_proxy,$ip_address
    docker compose -f comps/lvms/video-llama/server/docker/docker_compose_vllama.yaml up -d
    # docker run -d --name="test-comps-lvm-llava" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 8399:8399 --ipc=host opea/llava:latest
    # docker run -d --name="test-comps-lvm" -e LVM_ENDPOINT=http://$ip_address:8399 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 9399:9399 --ipc=host opea/lvm:latest
    sleep 8m # FIXME: change to detect if log has "Uvicorn running on"
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:9399/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        exit 1
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=lvm*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    # validate_microservice

    # stop_docker
    # echo y | docker system prune

}

main
