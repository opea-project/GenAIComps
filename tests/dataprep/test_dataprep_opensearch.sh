#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT="11104"
OPENSEARCH_INITIAL_ADMIN_PASSWORD="StRoNgOpEa0)"
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
    # Start OpenSearch dataprep container
    export OPENSEARCH_INITIAL_ADMIN_PASSWORD="StRoNgOpEa0)"
    export OPENSEARCH_PORT1=9200
    export OPENSEARCH_URL="http://${ip_address}:${OPENSEARCH_PORT1}"
    echo ${OPENSEARCH_URL}
    export INDEX_NAME="file-index"
    service_name="opensearch-vector-db dataprep-opensearch"
    export host_ip=${ip_address}
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate_microservice() {
    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    ingest_docx ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    ingest_txt ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    # test /v1/dataprep/ingest upload link
    ingest_external_link ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    # test /v1/dataprep/get
    get_all ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - get" '' dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log

    # test /v1/dataprep/delete
    delete_single ${ip_address} ${DATAPREP_PORT} -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD
    check_result "dataprep - del" '{"detail":"Single file deletion is not implemented yet"}' dataprep-opensearch-server ${LOG_PATH}/dataprep_opensearch.log "404"
}

function stop_service() {
    cid=$(docker ps -aq --filter "name=dataprep-opensearch-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    cid=$(docker ps -aq --filter "name=opensearch-vector-db")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {
    stop_service

    build_docker_images
    start_service

    validate_microservice

    stop_service
    echo y | docker system prune
}

main
