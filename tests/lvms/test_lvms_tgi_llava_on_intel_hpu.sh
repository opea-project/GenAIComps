#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export LLAVA_TGI_PORT=11502
export LVM_PORT=11503

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
    export LVM_ENDPOINT=http://$ip_address:$LLAVA_TGI_PORT
    export LLM_MODEL_ID="llava-hf/llava-v1.6-mistral-7b-hf"

    export LVM_COMPONENT_NAME=OPEA_TGI_LLAVA_LVM
    docker compose -f comps/lvms/deployment/docker_compose/compose.yaml up llava-tgi-service lvm-llava-tgi -d

    sleep 15s
}

function validate_microservice() {
    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "LVM prompt with an image - Result correct."
    else
        echo "LVM prompt with an image - Result wrong."
        docker logs llava-tgi-service >> ${LOG_PATH}/llava-tgi.log
        docker logs lvm-llava-tgi-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi

    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm --silent --write-out "HTTPSTATUS:%{http_code}" -XPOST -d '{"image": "", "prompt":"What is deep learning?"}' -H 'Content-Type: application/json')
    http_status=$(echo $result | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    if [ "$http_status" -ne "200" ]; then

        echo "LVM prompt without image - HTTP status is not 200. Received status was $http_status"
        docker logs llava-tgi-service >> ${LOG_PATH}/llava-tgi.log
        docker logs lvm-llava-tgi-service >> ${LOG_PATH}/lvm.log
        exit 1
    else
        echo "LVM prompt without image - HTTP status (successful)"
    fi

    # Test sending two images with a text prompt with one image tag in the prompt.
    # The first image is green and the second image is blue. Since the default MAX_IMAGES is 1, only the blue image should be sent to the LVM.
    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": ["iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC", "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC"], "prompt":"<image>\nWhat are in these images?"}' -H 'Content-Type: application/json')
    if [[ $result == *"blue"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs llava-tgi-service >> ${LOG_PATH}/llava-tgi.log
        docker logs lvm-llava-tgi-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi

    # Test sending two images with a text prompt without any image tags.
    # The first image is blue and the second image is green. Since the default MAX_IMAGES is 1, only the green image should be sent to the LVM.
    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": ["iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC", "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC"], "prompt":"What are in these images?"}' -H 'Content-Type: application/json')
    if [[ $result == *"green"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs llava-tgi-service >> ${LOG_PATH}/llava-tgi.log
        docker logs lvm-llava-tgi-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi

    # Same test as above, except including two image tags with the prompt to ensure the number of image tags is reconciled.
    # The first image is blue and the second image is green. Since the default MAX_IMAGES is 1, only the green image should be sent to the LVM.
    result=$(http_proxy="" curl http://localhost:$LVM_PORT/v1/lvm -XPOST -d '{"image": ["iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNkYPhfz0AEYBxVSF+FAP5FDvcfRYWgAAAAAElFTkSuQmCC", "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC"], "prompt":"<image>\n<image>\nWhat are in these images?"}' -H 'Content-Type: application/json')
    if [[ $result == *"green"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs llava-tgi-service >> ${LOG_PATH}/llava-tgi.log
        docker logs lvm-llava-tgi-service >> ${LOG_PATH}/lvm.log
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=llava-tgi-service" --filter "name=lvm-llava-tgi-service" --format "{{.Names}}" | xargs -r docker stop
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
