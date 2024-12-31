#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    
    # build dataprep image for pinecone
    docker build --no-cache -t opea/dataprep-pinecone:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f $WORKPATH/comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep-pinecone built fail"
        exit 1
    else
        echo "opea/dataprep-pinecone built successful"
    fi
}

function start_service() {
    # start dataprep service
    export PINECONE_API_KEY=$PINECONE_KEY
    export PINECONE_INDEX_NAME="test-index"
    export HUGGINGFACEHUB_API_TOKEN=$HF_TOKEN
    export dataprep_service_port=5039

    docker run -d --name="test-comps-dataprep-pinecone-server" -p ${dataprep_service_port}:5000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e PINECONE_API_KEY=$PINECONE_API_KEY -e PINECONE_INDEX_NAME=$PINECONE_INDEX_NAME -e LOGFLAG=true -e DATAPREP_TYPE="pinecone" opea/dataprep-pinecone:comps

    sleep 1m
}

function validate_service() {
    local URL="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_NAME="$3"
    local DOCKER_NAME="$4"
    local INPUT_DATA="$5"

    if [[ $SERVICE_NAME == *"dataprep_upload_file"* ]]; then
        cd $LOG_PATH
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -F 'files=@./dataprep_file.txt' -H 'Content-Type: multipart/form-data' "$URL")
    elif [[ $SERVICE_NAME == *"dataprep_upload_link"* ]]; then
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -F 'link_list=["https://www.ces.tech/"]' -F 'chunk_size=400' "$URL")
    elif [[ $SERVICE_NAME == *"dataprep_get"* ]]; then
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -H 'Content-Type: application/json' "$URL")
    elif [[ $SERVICE_NAME == *"dataprep_del"* ]]; then
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -d '{"file_path": "all"}' -H 'Content-Type: application/json' "$URL")
    else
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL")
    fi
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')

    docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_NAME}.log

    # check response status
    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"

        if [[ $SERVICE_NAME == *"dataprep_upload_link"* ]]; then
            docker logs test-comps-dataprep-milvus-tei-server >> ${LOG_PATH}/tei-embedding.log
        fi
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."
    fi
    # check response body
    if [[ "$RESPONSE_BODY" != *"$EXPECTED_RESULT"* ]]; then
        echo "[ $SERVICE_NAME ] Content does not match the expected result: $RESPONSE_BODY"
        exit 1
    else
        echo "[ $SERVICE_NAME ] Content is as expected."
    fi

    sleep 5s
}

function validate_microservice() {
    cd $LOG_PATH
    dataprep_service_port=5039

    # test /v1/dataprep/delete
    validate_service \
        "http://${ip_address}:${dataprep_service_port}/v1/dataprep/delete" \
        '{"status":true}' \
        "dataprep_del" \
        "test-comps-dataprep-pinecone-server"

    # test /v1/dataprep upload file
    echo "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data." > $LOG_PATH/dataprep_file.txt
    validate_service \
        "http://${ip_address}:${dataprep_service_port}/v1/dataprep/ingest" \
        "Data preparation succeeded" \
        "dataprep_upload_file" \
        "test-comps-dataprep-pinecone-server"

    # test /v1/dataprep upload link
    validate_service \
        "http://${ip_address}:${dataprep_service_port}/v1/dataprep/ingest" \
        "Data preparation succeeded" \
        "dataprep_upload_link" \
        "test-comps-dataprep-pinecone-server"

    # test /v1/dataprep/get_file
    validate_service \
        "http://${ip_address}:${dataprep_service_port}/v1/dataprep/get" \
        '{"name":' \
        "dataprep_get" \
        "test-comps-dataprep-pinecone-server"

}

function stop_docker() {
    cd $WORKPATH
    cid=$(docker ps -aq --filter "name=test-comps-dataprep-pinecone*")
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
