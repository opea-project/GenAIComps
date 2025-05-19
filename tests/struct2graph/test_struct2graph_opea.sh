#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

WORKPATH=$(git rev-parse --show-toplevel)
echo $WORKPATH
TAG='latest'
LOG_PATH="$WORKPATH/comps/struct2graph/deployment/docker_compose"
source $WORKPATH/comps/struct2graph/src/environment_setup.sh
STRUCT2GPAPH_PORT=8090
ip_address=$(hostname -I | awk '{print $1}')
service_name="struct2graph"

function build_docker_graph() {
    echo "===================  START BUILD DOCKER ========================"
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/struct2graph:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/struct2graph/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/struct2graph built fail"
        exit 1
    else
        echo "opea/struct2graph built successful"
    fi
    echo "===================  END BUILD DOCKER ========================"
}

function start_service() {
    echo "===================  START SERVICE ========================"
    cd $LOG_PATH
    docker compose -f struct2graph-compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 10s
    echo "===================  END SERVICE ========================"
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    cd $WORKPATH/tests/struct2graph
    python example_from_file.py
    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    echo "===================  START STOP DOCKER ========================"
    cd $LOG_PATH
    docker compose -f struct2graph-compose.yaml down --remove-orphans
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
