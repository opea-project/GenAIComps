#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT="11110"
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .

    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
    docker pull intellabs/vdms:latest
}

function start_service() {
    export host_ip=${ip_address}
    export VDMS_HOST=$ip_address
    export VDMS_PORT=55555
    export COLLECTION_NAME="test-comps"
    export QDRANT_HOST=$ip_address
    export QDRANT_PORT=$QDRANT_PORT
    service_name="vdms-vector-db dataprep-vdms"
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate_microservice() {
    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-vdms-server ${LOG_PATH}/dataprep_vdms.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-vdms-server ${LOG_PATH}/dataprep_vdms.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-vdms-server ${LOG_PATH}/dataprep_vdms.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-vdms-server ${LOG_PATH}/dataprep_vdms.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-vdms-server ${LOG_PATH}/dataprep_vdms.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-vdms-server ${LOG_PATH}/dataprep_vdms.log
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-vdms*" --filter "name=vdms-vector*")
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
