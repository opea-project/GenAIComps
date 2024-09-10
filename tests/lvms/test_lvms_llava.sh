#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/llava:comps -f comps/lvms/llava/dependency/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llava built fail"
        exit 1
    else
        echo "opea/llava built successful"
    fi
    docker build --no-cache -t opea/lvm:comps -f comps/lvms/llava/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/lvm built fail"
        exit 1
    else
        echo "opea/lvm built successful"
    fi
}

function start_service() {
    unset http_proxy
    llava_port=5071
    lvm_port=5072
    docker run -d --name="test-comps-lvm-llava-dependency" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p $lvm_port:8399 --ipc=host opea/llava:comps
    docker run -d --name="test-comps-lvm-llava-server" -e LVM_ENDPOINT=http://$ip_address:$llava_port -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p $lvm_port:9399 --ipc=host opea/lvm:comps
    sleep 8m
}

function validate_microservice() {

    lvm_port=5072
    result=$(http_proxy="" curl http://localhost:$lvm_port/v1/lvm -XPOST -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs test-comps-lvm-llava-dependency >> ${LOG_PATH}/llava-dependency.log
        docker logs test-comps-lvm-llava-server >> ${LOG_PATH}/llava-server.log
        exit 1
    fi

    result=$(http_proxy="" curl http://localhost:$lvm_port/v1/lvm -XPOST -d '{"retrieved_docs": [], "initial_query": "What is this?", "top_n": 1, "metadata": [{"b64_img_str": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "transcript_for_inference": "yellow image"}]}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs test-comps-lvm-llava-dependency >> ${LOG_PATH}/llava-dependency.log
        docker logs test-comps-lvm-llava-server >> ${LOG_PATH}/llava-server.log
        exit 1
    fi

    result=$(http_proxy="" curl http://localhost:$lvm_port/v1/lvm -XPOST -d '{"retrieved_docs": [], "initial_query": "What is this?", "top_n": 1, "metadata": [{"b64_img_str": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "transcript_for_inference": "yellow image"}], "chat_template":"The caption of the image is: '\''{context}'\''. {question}"}' -H 'Content-Type: application/json')
    if [[ $result == *"yellow"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs test-comps-lvm-llava-dependency >> ${LOG_PATH}/llava-dependency.log
        docker logs test-comps-lvm-llava-server >> ${LOG_PATH}/llava-server.log
        exit 1
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-lvm-llava*")
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
