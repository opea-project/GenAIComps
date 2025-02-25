#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT=11100
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

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
    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    # test /v1/dataprep/ingest upload link
    ingest_external_link ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    # test /v1/dataprep/get
    get_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - get" '{"name":' dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log

    # test /v1/dataprep/delete
    delete_single ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - del" '{"status":true}' dataprep-elasticsearch ${LOG_PATH}/dataprep_elastic.log
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
