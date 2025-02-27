#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT=11101
service_name="dataprep-milvus tei-embedding-serving etcd minio standalone"
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    # dataprep milvus image
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {
    export host_ip=${ip_address}
    export TEI_EMBEDDER_PORT=12005
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export MILVUS_HOST=${ip_address}
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export LOGFLAG=true

    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 1m
}

function validate_microservice() {
    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    # test /v1/dataprep/ingest upload link
    ingest_external_link ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    # test /v1/dataprep/get
    get_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - get" '{"name":' dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    # test /v1/dataprep/delete
    delete_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - del" '{"status":true}' dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log
}

function stop_docker() {
    cd $WORKPATH/comps/third_parties/milvus/deployment/docker_compose/
    docker compose -f compose.yaml down --remove-orphans

    cd $WORKPATH/comps/dataprep/deployment/docker_compose
    docker compose -f compose.yaml down  ${service_name} --remove-orphans

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
