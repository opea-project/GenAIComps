#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT=11100
export TAG="comps"

function build_docker_images() {
    cd $WORKPATH
    echo $WORKPATH
    # piull elasticsearch image
    docker pull docker.elastic.co/elasticsearch/elasticsearch:8.16.0

    # build dataprep image for elasticsearch
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f $WORKPATH/comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export ELASTICSEARCH_PORT1=12300
    export ES_CONNECTION_STRING="http://${ip_address}:${ELASTICSEARCH_PORT1}"
    export INDEX_NAME="test-elasticsearch"
    service_name="elasticsearch-vector-db dataprep-elasticsearch"
    cd $WORKPATH
    cd comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
    echo "Microservice started"
}

function validate_microservice() {
    cd $LOG_PATH

    # test /v1/dataprep
    URL="http://${ip_address}:$DATAPREP_PORT/v1/dataprep/ingest"
    echo "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data." > $LOG_PATH/dataprep_file.txt

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -F 'files=@./dataprep_file.txt' -H 'Content-Type: multipart/form-data' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ dataprep ] HTTP status is 200. Checking content..."
        cp ./dataprep_file.txt ./dataprep_file2.txt
        local CONTENT=$(curl -s -X POST -F 'files=@./dataprep_file2.txt' -H 'Content-Type: multipart/form-data' "$URL" | tee ${LOG_PATH}/dataprep.log)

        if echo "$CONTENT" | grep -q "Data preparation succeeded"; then
            echo "[ dataprep ] Content is as expected."
        else
            echo "[ dataprep ] Content does not match the expected result: $CONTENT"
            docker logs dataprep-elasticsearch >> ${LOG_PATH}/dataprep.log
            exit 1
        fi
    else
        echo "[ dataprep ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs dataprep-elasticsearch >> ${LOG_PATH}/dataprep.log
        exit 1
    fi

    # test /v1/dataprep/get_file
    URL="http://${ip_address}:$DATAPREP_PORT/v1/dataprep/get"
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ dataprep - file ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/dataprep_file.log)

        if echo "$CONTENT" | grep -q '{"name":'; then
            echo "[ dataprep - file ] Content is as expected."
        else
            echo "[ dataprep - file ] Content does not match the expected result: $CONTENT"
            docker logs dataprep-elasticsearch >> ${LOG_PATH}/dataprep_file.log
            exit 1
        fi
    else
        echo "[ dataprep - file ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs dataprep-elasticsearch >> ${LOG_PATH}/dataprep_file.log
        exit 1
    fi

    # test /v1/dataprep/delete_file
    URL="http://${ip_address}:$DATAPREP_PORT/v1/dataprep/delete"
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d '{"file_path": "dataprep_file.txt"}' -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ dataprep - del ] HTTP status is 200."
        docker logs dataprep-elasticsearch >> ${LOG_PATH}/dataprep_del.log
    else
        echo "[ dataprep - del ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs dataprep-elasticsearch >> ${LOG_PATH}/dataprep_del.log
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=elasticsearch-vector-db" --filter "name=dataprep-elasticsearch")
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
