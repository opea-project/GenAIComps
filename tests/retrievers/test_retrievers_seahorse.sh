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

source "${WORKPATH}/tests/utils/seahorse_helpers.sh"
require_seahorse_credentials_or_skip
detect_host_ip
service_name="retriever-seahorse"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/retriever:${TAG:-latest} \
        --build-arg https_proxy=$https_proxy \
        --build-arg http_proxy=$http_proxy \
        -f comps/retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever built fail"
        exit 1
    else
        echo "opea/retriever built successful"
    fi
}

function wait_for_endpoint() {
    local url=$1
    local retries=30
    local count=0

    echo "Waiting for ${url} to become reachable..."
    while [ $count -lt $retries ]; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "${url} is reachable!"
            return 0
        fi
        echo "  → ${url} not ready yet ($count/$retries)"
        sleep 5
        ((count++))
    done

    echo "${url} did not respond in time."
    return 1
}

function start_service() {
    export RETRIEVER_PORT=7000
    export LOGFLAG=True

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d \
        > ${LOG_PATH}/start_services_with_compose.log

    wait_for_endpoint "http://${host_ip}:${RETRIEVER_PORT}/v1/health_check" || {
        docker logs ${service_name} >> ${LOG_PATH}/retriever.log
        exit 1
    }
}

function _retriever_post() {
    local url="http://${host_ip}:${RETRIEVER_PORT}/v1/retrieval"
    local payload=$1

    HTTP_RESPONSE=$(curl --silent --show-error --write-out "\nHTTPSTATUS:%{http_code}" \
        -X POST -H 'Content-Type: application/json' -d "${payload}" "${url}")
    RESPONSE_BODY=$(echo "$HTTP_RESPONSE" | sed '$d')
    HTTP_STATUS=$(echo "$HTTP_RESPONSE" | tr -d '\n' | sed -n 's/.*HTTPSTATUS:\([0-9]*\)$/\1/p')
}

function _expect_status() {
    local label=$1
    local expected=$2

    if [ "${HTTP_STATUS}" -ne "${expected}" ]; then
        echo "[ ${label} ] Expected HTTP ${expected}, got ${HTTP_STATUS}. Body=${RESPONSE_BODY}"
        docker logs ${service_name} >> ${LOG_PATH}/retriever.log
        exit 1
    fi
    echo "[ ${label} ] HTTP status ${HTTP_STATUS} as expected."
}

function _expect_body_contains() {
    local label=$1
    local needle=$2

    if ! echo "${RESPONSE_BODY}" | grep -q "${needle}"; then
        echo "[ ${label} ] Body did not contain '${needle}'. Body=${RESPONSE_BODY}"
        docker logs ${service_name} >> ${LOG_PATH}/retriever.log
        exit 1
    fi
    echo "[ ${label} ] Body contains '${needle}'."
}

function validate_microservice() {
    # 1) Plain similarity search (builtin: text only, embedding placeholder).
    _retriever_post '{"text":"test query","embedding":[0.1],"k":3}'
    _expect_status "retriever - similarity" 200
    _expect_body_contains "retriever - similarity" "retrieved_docs"

    # 2) Distance threshold path (dense semantics; result list may be empty if the
    #    table holds no matches above threshold, but the service must still 200).
    _retriever_post '{"text":"test query","embedding":[0.1],"k":3,"search_type":"similarity_distance_threshold","distance_threshold":1.0}'
    _expect_status "retriever - distance_threshold" 200
    _expect_body_contains "retriever - distance_threshold" "retrieved_docs"

    # 3) Unsupported search_type must surface a 4xx (validated by _normalize_search_type).
    _retriever_post '{"text":"test query","embedding":[0.1],"k":3,"search_type":"bogus_search"}'
    if [ "${HTTP_STATUS}" -ge 200 ] && [ "${HTTP_STATUS}" -lt 300 ]; then
        echo "[ retriever - bogus search_type ] Expected error, got HTTP ${HTTP_STATUS}. Body=${RESPONSE_BODY}"
        docker logs ${service_name} >> ${LOG_PATH}/retriever.log
        exit 1
    fi
    echo "[ retriever - bogus search_type ] Rejected as expected (HTTP ${HTTP_STATUS})."
}

function stop_docker() {
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
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
