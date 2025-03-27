#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"
export DATA_PATH=${model_cache}

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
export host_ip=$(hostname -I | awk '{print $1}')
service_name="opensearch-vector-db tei-embedding-serving retriever-opensearch"
retriever_service_name="retriever-opensearch"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/retriever:${TAG:-latest}  --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever built fail"
        exit 1
    else
        echo "opea/retriever built successful"
    fi
}

function start_service() {
    export OPENSEARCH_PORT1=11627
    export OPENSEARCH_PORT2=11628
    export RETRIEVER_PORT=11629
    export TEI_EMBEDDER_PORT=11630
    export OPENSEARCH_INITIAL_ADMIN_PASSWORD="StRoNgOpEa0)"
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
    export OPENSEARCH_URL="http://${host_ip}:${OPENSEARCH_PORT1}"
    export INDEX_NAME="file-index"

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log


    sleep 2m
}

function validate_microservice() {
    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    URL="http://${host_ip}:$RETRIEVER_PORT/v1/retrieval"

    test_embedding=$(python3  -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' -k -u admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content is as expected."
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs ${retriever_service_name} >> ${LOG_PATH}/retriever.log
            docker logs tei-embedding-serving >> ${LOG_PATH}/tei.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs ${retriever_service_name} >> ${LOG_PATH}/retriever.log
        docker logs tei-embedding-serving >> ${LOG_PATH}/tei.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
    cid=$(docker ps -aq --filter "name=opensearch-vector-db" --filter "name=tei-embedding-serving")
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
