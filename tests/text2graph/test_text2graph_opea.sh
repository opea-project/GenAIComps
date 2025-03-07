#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x
WORKPATH=$(git rev-parse --show-toplevel)
TAG='latest'
LOG_PATH="$WORKPATH/comps/text2graph/deployment/docker_compose"
source $WORKPATH/comps/text2graph/src/setup_service_env.sh


echo $WORKPATH
ip_address=$(hostname -I | awk '{print $1}')
service_name="text2graph"

function build_docker_graph() {
    echo "===================  START BUILD DOCKER ========================"
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/test2graph:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2graph/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2graph built fail"
        exit 1
    else
        echo "opea/text2graph built successful"
    fi
    echo "===================  END BUILD DOCKER ========================"
}

function start_service() {
    echo "===================  START SERVICE ========================"
    cd $WORKPATH/comps/text2graph/deployment/docker_compose
    docker compose up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 10s
    echo "===================  END SERVICE ========================"
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    cd $WORKPATH/tests/text2graph
    python3 example_from_file.py
    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    echo "===================  START STOP DOCKER ========================"
    cd $WORKPATH/comps/text2graph/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
    echo "===================  END STOP DOCKER ========================"
}

function main() {

    stop_docker

    build_docker_graph
    start_service
    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
