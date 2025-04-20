#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"
export DATA_PATH=${model_cache}

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
export host_ip=$(hostname -I | awk '{print $1}')
service_name="retriever-vdms"
service_name_mm="retriever-vdms-multimodal"
export no_proxy="${no_proxy},${host_ip},${service_name},${service_name_mm}"
export HF_TOKEN=${HF_TOKEN}
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export VDMS_PORT=11624
export RETRIEVER_PORT=11625
export RETRIEVER_COMPONENT_NAME="OPEA_RETRIEVER_VDMS"
export LOGFLAG=True

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/retriever:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever built fail"
        exit 1
    else
        echo "opea/retriever built successful"
    fi

}

function start_service() {
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export INDEX_NAME="rag-vdms"
    export TEI_EMBEDDER_PORT=11626
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export VDMS_USE_CLIP=0 #set to 1 if openai clip embedding should be used

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 1m
}

function start_multimodal_service() {
    export INDEX_NAME="mm-rag-vdms"
    export VDMS_USE_CLIP=1

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name_mm} -d > ${LOG_PATH}/start_services_with_compose_multimodal.log

    sleep 1m
}

function validate_services() {
    local URL="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_SHORT_NAME="$3"
    local DOCKER_NAME="$4"
    local INPUT_DATA="$5"

    HTTP_RESPONSE=$(http_proxy="" curl -v -w "HTTPSTATUS:%{http_code}" -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL")
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')

    echo "HTTP_RESPONSE=${HTTP_RESPONSE}"

    docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_SHORT_NAME}.log

    # check response status
    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_SHORT_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        exit 1
    else
        echo "[ $SERVICE_SHORT_NAME ] HTTP status is 200. Checking content..."
    fi

    # check response body
    if [[ "${RESPONSE_BODY}" != *"${EXPECTED_RESULT}"* ]]; then
        echo "[ $SERVICE_SHORT_NAME ] Content does not match the expected result: $RESPONSE_BODY" >> ${LOG_PATH}/${SERVICE_SHORT_NAME}.log
        exit 1
    else
        echo "[ $SERVICE_SHORT_NAME ] Content is as expected: $RESPONSE_BODY" >> ${LOG_PATH}/${SERVICE_SHORT_NAME}.log
    fi
    sleep 1s
}

function validate_microservice() {
    # Retriever Microservice
    test_embedding=$(python3 -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
    validate_services \
        "http://${host_ip}:$RETRIEVER_PORT/v1/retrieval" \
        "retrieved_docs" \
        ${service_name} \
        ${service_name} \
        "{\"text\":\"Sample text\",\"embedding\":${test_embedding},\"search_type\":\"similarity\"}"
}

function validate_mm_microservice() {
    # Multimodal Retriever Microservice
    test_embedding_multi=$(python3 -c "import random; embedding = [random.uniform(-1, 1) for _ in range(512)]; print(embedding)")
    validate_services \
        "http://0.0.0.0:$RETRIEVER_PORT/v1/retrieval" \
        "retrieved_docs" \
        ${service_name_mm} \
        ${service_name_mm} \
        "{\"text\":\"Sample text\",\"embedding\":${test_embedding_multi},\"search_type\":\"mmr\"}"
}

function stop_docker() {
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
    cid=$(docker ps -aq --filter "name=retriever-vdms*" --filter "name=vdms-vector-db" --filter "name=tei-embedding-serving")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images

    start_service
    validate_microservice
    stop_docker

    start_multimodal_service
    validate_mm_microservice

    stop_docker
    echo y | docker system prune

}

main
