#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export LLAMA_VISION_PORT=11510
export LVM_PORT=11511

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/lvm:$TAG -f comps/lvms/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/lvm built fail"
        exit 1
    else
        echo "opea/lvm built successful"
    fi
}

function start_service() {

    unset http_proxy
    export LVM_ENDPOINT=http://$ip_address:$LLAMA_VISION_PORT
    export LLM_MODEL_ID="meta-llama/Llama-3.2-11B-Vision-Instruct"

    export LVM_COMPONENT_NAME=OPEA_LLAMA_VISION_LVM
    docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up llama-vision-service lvm-llama-vision -d

    sleep 15s
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "LVM prompt with an image - Result correct."
    else
        echo "LVM prompt with an image - Result wrong."
        docker logs llama-vision-service >> ${LOG_PATH}/llama-vision.log
        docker logs lvm-llama-vision-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi

    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm --silent --write-out "HTTPSTATUS:%{http_code}" -XPOST -d '{"image": "", "prompt":"What is deep learning?"}' -H 'Content-Type: application/json')
    http_status=$(echo $result | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    if [ "$http_status" -ne "200" ]; then

        echo "LVM prompt without image - HTTP status is not 200. Received status was $http_status"
        docker logs llama-vision-service >> ${LOG_PATH}/llama-vision.log
        docker logs lvm-llama-vision-service >> ${LOG_PATH}/lvm.log
        exit 1
    else
        echo "LVM prompt without image - HTTP status (successful)"
    fi

}

function stop_docker() {
    docker ps -a --filter "name=llama-vision-service" --filter "name=lvm-llama-vision-service" --format "{{.Names}}" | xargs -r docker stop
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
