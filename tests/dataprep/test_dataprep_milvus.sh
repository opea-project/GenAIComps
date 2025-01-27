#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT=11101

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    # dataprep milvus image
    docker build --no-cache -t opea/dataprep:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {
    # start milvus vector db
    cd $WORKPATH/comps/third_parties/milvus/deployment/docker_compose/
    # wget https://raw.githubusercontent.com/milvus-io/milvus/v2.4.9/configs/milvus.yaml
    # wget https://github.com/milvus-io/milvus/releases/download/v2.4.9/milvus-standalone-docker-compose.yml -O docker-compose.yml
    # sed '/- \${DOCKER_VOLUME_DIRECTORY:-\.}\/volumes\/milvus:\/var\/lib\/milvus/a \ \ \ \ \ \ - \${DOCKER_VOLUME_DIRECTORY:-\.}\/milvus.yaml:\/milvus\/configs\/milvus.yaml' -i docker-compose.yml
    docker compose up -d
    sleep 30

    export host_ip=${ip_address}
    export TEI_EMBEDDER_PORT=12005
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export MILVUS_HOST=${ip_address}
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    service_name="dataprep-milvus tei-embedding-serving"
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
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

    # test /v1/dataprep/delete
    validate_service \
        "http://${ip_address}:${DATAPREP_PORT}/v1/dataprep/delete" \
        '{"status":true}' \
        "dataprep_del" \
        "dataprep-milvus-server"

    # test /v1/dataprep upload file
    echo "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data." > $LOG_PATH/dataprep_file.txt
    validate_service \
        "http://${ip_address}:${DATAPREP_PORT}/v1/dataprep/ingest" \
        "Data preparation succeeded" \
        "dataprep_upload_file" \
        "dataprep-milvus-server"

    # test /v1/dataprep upload link
    validate_service \
        "http://${ip_address}:${DATAPREP_PORT}/v1/dataprep/ingest" \
        "Data preparation succeeded" \
        "dataprep_upload_link" \
        "dataprep-milvus-server"

    # test /v1/dataprep/get_file
    validate_service \
        "http://${ip_address}:${DATAPREP_PORT}/v1/dataprep/get" \
        '{"name":' \
        "dataprep_get" \
        "dataprep-milvus-server"

}

function stop_docker() {
    cd $WORKPATH
    rm -rf milvus/
    cid=$(docker ps -aq --filter "name=dataprep-milvus*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    cid=$(docker ps -aq --filter "name=milvus-*")
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
