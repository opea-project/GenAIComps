#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export VDMS_PORT=55555

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    # pull vdms image
    docker pull intellabs/vdms:latest

    # build dataprep image for vdms
    docker build -t opea/dataprep-vdms:comps \
        --build-arg https_proxy=$https_proxy \
        --build-arg http_proxy=$http_proxy \
        -f comps/dataprep/vdms/langchain/docker/Dockerfile .
}

function start_service() {
    docker run -d --name "test-comps-dataprep-vdms-langchain" -p $VDMS_PORT:55555 --ipc=host \
        -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
        intellabs/vdms:latest

    sleep 10s

    dataprep_service_port=5020
    docker run -d --name="test-comps-dataprep-vdms-langchain-server" -p ${dataprep_service_port}:6007 --ipc=host \
        -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TEI_ENDPOINT=$TEI_ENDPOINT \
        -e VDMS_HOST=$ip_address -e VDMS_PORT=$VDMS_PORT \
        opea/dataprep-vdms:comps

    sleep 30s
}

# function validate_microservice() {
#     cd $LOG_PATH
#     dataprep_service_port=6007  #5020
#     URL="http://$ip_address:$dataprep_service_port/v1/dataprep"
#     echo 'The OPEA platform includes: Detailed framework of composable building blocks for state-of-the-art generative AI systems including LLMs, data stores, and prompt engines' > $LOG_PATH/dataprep_file.txt
#     curl --noproxy $ip_address --location --request POST \
#       --form 'files=@$LOG_PATH/dataprep_file.txt' $URL
# }

function validate_microservice() {
    cd $LOG_PATH
    dataprep_service_port=5020
    URL="http://${ip_address}:$dataprep_service_port/v1/dataprep"
    echo "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data." > $LOG_PATH/dataprep_file.txt
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -F 'files=@./dataprep_file.txt' -H 'Content-Type: multipart/form-data' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ dataprep ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -F 'files=@./dataprep_file.txt' -H 'Content-Type: multipart/form-data' "$URL" | tee ${LOG_PATH}/dataprep.log)

        if echo 'Data preparation succeeded' | grep -q "$EXPECTED_RESULT"; then
            echo "[ dataprep ] Content is as expected."
        else
            echo "[ dataprep ] Content does not match the expected result: $CONTENT"
            docker logs test-comps-dataprep-vdms-langchain-server >> ${LOG_PATH}/dataprep.log
            exit 1
        fi
    else
        echo "[ dataprep ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs test-comps-dataprep-vdms-langchain-server >> ${LOG_PATH}/dataprep.log
        exit 1
    fi
    rm -rf $LOG_PATH/dataprep_file.txt
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-dataprep-vdms-langchain*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi

    cid=$(docker ps -aq --filter "name=test-comps-dataprep-vdms-langchain-server*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    # stop_docker
    # echo y | docker system prune

}

main
