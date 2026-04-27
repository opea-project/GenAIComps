#!/bin/bash
# Copyright (C) 2026 Dnotitia
# SPDX-License-Identifier: Apache-2.0

set -xe

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKPATH=$(cd "${SCRIPT_DIR}/../.." && pwd)
LOG_PATH="$WORKPATH/tests"

source "${SCRIPT_DIR}/dataprep_utils.sh"
source "${WORKPATH}/tests/utils/seahorse_helpers.sh"
require_seahorse_credentials_or_skip
detect_host_ip
service_name="dataprep-seahorse-server"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/dataprep:${TAG:-latest} \
        --build-arg https_proxy=$https_proxy \
        --build-arg http_proxy=$http_proxy \
        -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {
    export DATAPREP_PORT=5000
    export LOGFLAG=True

    cd $WORKPATH/comps/dataprep/deployment/docker_compose
    docker compose -f compose.yaml up dataprep-seahorse -d \
        > ${LOG_PATH}/start_services_with_compose.log

    check_healthy "${service_name}" || exit 1
}

function validate_microservice() {
    # Start clean so per-format counts are deterministic.
    delete_all "${host_ip}" "${DATAPREP_PORT}"
    check_result "dataprep - reset" '{"status":true}' "${service_name}" ${LOG_PATH}/dataprep_del.log

    # txt
    ingest_txt "${host_ip}" "${DATAPREP_PORT}" "seahorse"
    check_result "dataprep - upload - txt" "Data preparation succeeded" "${service_name}" ${LOG_PATH}/dataprep_upload_file.log

    # docx
    ingest_docx "${host_ip}" "${DATAPREP_PORT}" "seahorse"
    check_result "dataprep - upload - docx" "Data preparation succeeded" "${service_name}" ${LOG_PATH}/dataprep_upload_file.log

    # pdf
    ingest_pdf "${host_ip}" "${DATAPREP_PORT}" "seahorse"
    check_result "dataprep - upload - pdf" "Data preparation succeeded" "${service_name}" ${LOG_PATH}/dataprep_upload_file.log

    # external link (skip in fully air-gapped environments by setting SKIP_LINK_TEST=1)
    if [[ "${SKIP_LINK_TEST:-0}" != "1" ]]; then
        ingest_external_link "${host_ip}" "${DATAPREP_PORT}"
        check_result "dataprep - upload - link" "Data preparation succeeded" "${service_name}" ${LOG_PATH}/dataprep_upload_file.log
    fi

    # get the local file index returned by the dataprep service
    get_all "${host_ip}" "${DATAPREP_PORT}"
    check_result "dataprep - get" '"name":' "${service_name}" ${LOG_PATH}/dataprep_file.log

    # delete a single file by path (txt fixture is named ingest_dataprep.txt)
    delete_single "${host_ip}" "${DATAPREP_PORT}"
    check_result "dataprep - del single" '{"status":true}' "${service_name}" ${LOG_PATH}/dataprep_del.log

    # delete everything
    delete_all "${host_ip}" "${DATAPREP_PORT}"
    check_result "dataprep - del all" '{"status":true}' "${service_name}" ${LOG_PATH}/dataprep_del.log
}

function stop_docker() {
    cd $WORKPATH/comps/dataprep/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
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
