#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT="11108"
TEI_EMBEDDER_PORT="10221"
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {

    export host_ip=${ip_address}
    export REDIS_HOST=$ip_address
    export REDIS_PORT=6379
    export DATAPREP_PORT="11108"
    export TEI_EMBEDDER_PORT="10221"
    export REDIS_URL="redis://${ip_address}:${REDIS_PORT}"
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${TEI_EMBEDDER_PORT}"
    export INDEX_NAME="rag_redis"
    service_name="redis-vector-db tei-embedding-serving dataprep-redis"
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate_microservice() {

    # test /v1/dataprep/delete
    delete_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - del" '{"status":true}' dataprep-redis-server ${LOG_PATH}/dataprep_del.log

    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    # test /v1/dataprep/ingest upload link
    ingest_external_link ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-redis-server ${LOG_PATH}/dataprep_upload_file.log

    # test /v1/dataprep/get
    get_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - get" '{"name":' dataprep-redis-server ${LOG_PATH}/dataprep_file.log
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-redis-server*" --filter "name=redis-vector-*" --filter "name=tei-embedding-*")
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
