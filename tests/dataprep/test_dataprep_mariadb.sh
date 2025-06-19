#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export DATAPREP_PORT="11105"
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH

    # build dataprep image for mariadb
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f $WORKPATH/comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {
    export host_ip=${ip_address}
    export EMBEDDING_LENGTH=768
    export MARIADB_PORT=11617
    export DATAPREP_PORT=11618
    export MARIADB_USER=testuser
    export MARIADB_PASSWORD=testpwd
    export MARIADB_DATABASE=vectordb
    export MARIADB_CONNECTION_URL=mariadb+mariadbconnector://${MARIADB_USER}:${MARIADB_PASSWORD}@$host_ip:$MARIADB_PORT/${MARIADB_DATABASE}
    export LOGFLAG=True

    service_name="dataprep-mariadb-vector"

    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d

    check_healthy "dataprep-mariadb-vector" || exit 1
}

function validate_microservice() {
    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    ingest_ppt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - ppt" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_upload_file.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    # test /v1/dataprep/ingest upload link
    ingest_external_link ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    # test /v1/dataprep/get
    get_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - get" '{"name":' dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log

    # test /v1/dataprep/delete
    delete_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - del" '{"status":true}' dataprep-mariadb-vector ${LOG_PATH}/dataprep_mariadb.log
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-mariadb-vector")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi

    cid=$(docker ps -aq --filter "name=mariadb-server")
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
