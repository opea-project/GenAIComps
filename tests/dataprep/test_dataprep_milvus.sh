#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export DATAPREP_PORT=11101
service_name="dataprep-milvus tei-embedding-serving etcd minio standalone"
export TAG="comps"
export DATA_PATH=${model_cache}
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
    local offline=${1:-false}
    export host_ip=${ip_address}
    export TEI_EMBEDDER_PORT=12005
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export MILVUS_HOST=${ip_address}
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export LOGFLAG=true

    if [[ "$offline" == "true" ]]; then
        service_name="dataprep-milvus-offline tei-embedding-serving etcd minio standalone"
        export offline_no_proxy="${ip_address},${host_ip}"
    else
        service_name="dataprep-milvus tei-embedding-serving etcd minio standalone"
    fi
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    check_healthy "dataprep-milvus-server" || exit 1
}

function validate_microservice() {
    local offline=${1:-false}
    # test /v1/dataprep/delete
    delete_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - del" '{"status":true}' dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_ppt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - ppt" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log

    # test /v1/dataprep/ingest upload link
    if [[ "$offline" != "true" ]]; then
      ingest_external_link ${ip_address} ${DATAPREP_PORT}
      check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-milvus-server ${LOG_PATH}/dataprep_milvus.log
    fi

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
    docker compose -f compose.yaml down --remove-orphans

}

function main() {

    stop_docker

    build_docker_images
    trap stop_docker EXIT

    echo "Test normal env ..."
    start_service
    validate_microservice
    stop_docker

    if [[ -n "${DATA_PATH}" ]]; then
        echo "Test air gapped env ..."
        prepare_dataprep_models ${DATA_PATH}
        start_service true
        validate_microservice true
        stop_docker
    fi

    echo y | docker system prune

}

main
