#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x
WORKPATH=$(git rev-parse --show-toplevel)
TAG='latest'
LOG_PATH="$WORKPATH/comps/text2kg/deployment/docker_compose"
source $WORKPATH/comps/text2kg/src/environment_setup.sh


echo $WORKPATH
ip_address=$(hostname -I | awk '{print $1}')
service_name="text2kg"

function build_docker() {
    echo "===================  START BUILD DOCKER ========================"
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2kg:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2kg/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/text2kg built fail"
        exit 1
    else
        echo "opea/text2kg built successful"
    fi
    echo "===================  END BUILD DOCKER ========================"
}

function start_service() {
    echo "===================  START SERVICE ========================"
    cd $WORKPATH/comps/text2kg/deployment/docker_compose
    docker compose -f compose.yaml -f custom-override.yml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 10s
    echo "===================  END SERVICE ========================"
}

function validate_microservice() {
    echo "===================  START VALIDATE ========================"
    cd $WORKPATH/tests/text2kg

    # Download test file
    FILE_URL="https://gist.githubusercontent.com/wey-gu/75d49362d011a0f0354d39e396404ba2/raw/0844351171751ebb1ce54ea62232bf5e59445bb7/paul_graham_essay.txt"
    wget -P "$DATA_DIRECTORY" "$FILE_URL"

    if wget -P "$DATA_DIRECTORY" "$FILE_URL"; then
        echo "Download successful"
    else
        echo "Download failed"
        return 1
    fi

    # Test API endpoint
    result=$(curl -X POST \
          -H "accept: application/json" \
          -d "" \
          http://localhost:8090/v1/text2kg?input_text=Who%20is%20paul%20graham%3F)

    if [[ $result == *"output"* ]]; then
        echo $result
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs text2kg > ${LOG_PATH}/text2kg.log
        return 1
    fi

    echo "===================  END VALIDATE ========================"
}

function stop_docker() {
    echo "===================  START STOP DOCKER ========================"
    cd $WORKPATH/comps/text2kg/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
    echo "===================  END STOP DOCKER ========================"
}

function main() {

    stop_docker

    build_docker
    start_service
    validate_microservice

    stop_docker

}

main
