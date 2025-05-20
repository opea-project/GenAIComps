#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export LLAVA_PORT=11512
export LVM_PORT=11513
export ENABLE_MCP=True
cd $WORKPATH

function build_docker_images() {
    echo $(pwd)
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm-llava:$TAG -f comps/third_parties/llava/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/lvm-llava built fail"
        exit 1
    else
        echo "opea/lvm-llava built successful"
    fi
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm:$TAG -f comps/lvms/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/lvm built fail"
        exit 1
    else
        echo "opea/lvm built successful"
    fi
}

function start_service() {

    export LVM_ENDPOINT=http://$ip_address:$LLAVA_PORT

    export LVM_COMPONENT_NAME=OPEA_LLAVA_LVM
    docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up llava-service lvm-llava -d
    sleep 1m
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/utils/validate_svc_with_mcp.py $ip_address $LVM_PORT "lvm"
    if [ $? -ne 0 ]; then
        docker logs llava-service
        docker logs lvm-llava-service
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=llava-service" --filter "name=lvm-llava-service" --format "{{.Names}}" | xargs -r docker stop
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
