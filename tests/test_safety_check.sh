#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/security:latest -f comps/security/Dockerfile .
}

function start_service() {
    unset http_proxy
    tei_service_port=6008
    docker run -d --name="test-comps-security-server" -p ${tei_service_port}:6008 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/security:latest
    sleep 10s
}

function validate_microservice() {
    tei_service_port=6008
    http_proxy="" curl http://${ip_address}:${tei_service_port}/v1/safety/check \
        -X POST \
        -H "Content-Type: application/json"   \
        -d '{"text":"bomb the interchange","path":"/home/user/comps/security"}'
    docker logs test-comps-security-server
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-security*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
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
