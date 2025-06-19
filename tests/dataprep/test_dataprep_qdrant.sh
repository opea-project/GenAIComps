#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export DATAPREP_PORT="11107"
TEI_EMBEDDER_PORT="10220"
export TAG="comps"
export DATA_PATH=${model_cache}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH

    # dataprep qdrant image
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
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export EMBED_MODEL=${EMBEDDING_MODEL_ID}
    export TEI_EMBEDDER_PORT="10224"
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${TEI_EMBEDDER_PORT}"
    export COLLECTION_NAME="rag-qdrant"
    export QDRANT_HOST=$ip_address
    export QDRANT_PORT=6360
    if [[ "$offline" == "true" ]]; then
        service_name="qdrant-vector-db tei-embedding-serving dataprep-qdrant-offline"
        export offline_no_proxy="${ip_address}"
    else
        service_name="qdrant-vector-db tei-embedding-serving dataprep-qdrant"
    fi
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d

    check_healthy "dataprep-qdrant-server" || exit 1
}

function validate_microservice() {
    local offline=${1:-false}
    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log

    ingest_ppt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - ppt" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep_upload_file.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log

    # test /v1/dataprep/ingest upload link
    if [[ "$offline" != "true" ]]; then
      ingest_external_link ${ip_address} ${DATAPREP_PORT}
      check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-qdrant-server ${LOG_PATH}/dataprep-qdrant.log
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-qdrant-server*" --filter "name=tei-embedding-serving*" --filter "name=qdrant-vector-db")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function stop_service() {
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose down || true
}

function main() {

    stop_docker

    build_docker_images
    trap stop_service EXIT

    echo "Test normal env ..."
    start_service
    validate_microservice
    stop_service

    if [[ -n "${DATA_PATH}" ]]; then
        echo "Test air gapped env ..."
        prepare_dataprep_models ${DATA_PATH}
        start_service true
        validate_microservice true
        stop_service
    fi

    stop_docker
    echo y | docker system prune

}

main
