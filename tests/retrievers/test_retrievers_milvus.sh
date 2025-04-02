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
service_name="retriever-milvus etcd minio standalone"
retriever_service_name="retriever-milvus"
service_name_mm="retriever-milvus-multimodal etcd minio standalone"

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
    export MINIO_PORT1=11611
    export MINIO_PORT2=11612
    export MILVUS_STANDALONE_PORT=11613
    export TEI_EMBEDDER_PORT=11614
    export RETRIEVER_PORT=11615
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export HF_TOKEN=${HF_TOKEN}
    export LOGFLAG=True
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export MILVUS_HOST=${host_ip}

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 1m
}

function start_multimodal_service() {
    export MINIO_PORT1=11611
    export MINIO_PORT2=11612
    export MILVUS_STANDALONE_PORT=11613
    export TEI_EMBEDDER_PORT=11614
    export HF_TOKEN=${HF_TOKEN}
    export COLLECTION_NAME="LangChainCollection"
    export LOGFLAG=True
    export BRIDGE_TOWER_EMBEDDING=true
    export RETRIEVER_TYPE="milvus"

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name_mm} -d > ${LOG_PATH}/start_services_with_compose_multimodal.log

    sleep 2m
}

function validate_microservice() {
    local test_embedding="$1"

    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    URL="http://${host_ip}:$RETRIEVER_PORT/v1/retrieval"

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content is as expected."
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs ${retriever_service_name} >> ${LOG_PATH}/retriever.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs ${retriever_service_name} >> ${LOG_PATH}/retriever.log
        exit 1
    fi
}

function validate_mm_microservice() {
    local test_embedding="$1"
    local container_name="$2"

    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    URL="http://${host_ip}:$RETRIEVER_PORT/v1/retrieval"

    # Test the retriever with a b64 image that should be passed through
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding},\"base64_image\":\"iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC\"}" -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content has retrieved_docs as expected."
            empty_search_results=$(echo "$CONTENT" | grep "\"retrieved_docs\":\[\]")
            if [ -z "$empty_search_results" ]; then
                # If search results are not empty, check for b64 image string
                if echo "$CONTENT" | grep -q "b64_img_str"; then
                    echo "[ retriever ] Content has b64_img_str as expected."
                else
                    echo "[ retriever ] Content does not include the b64_img_str: $CONTENT"
                    docker logs ${container_name} >> ${LOG_PATH}/retriever.log
                    exit 1
                fi
            fi
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs ${container_name} >> ${LOG_PATH}/retriever.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs ${container_name} >> ${LOG_PATH}/retriever.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/third_parties/milvus/deployment/docker_compose/
    docker compose -f compose.yaml down --remove-orphans

    docker stop milvus-etcd
    docker rm milvus-etcd

    cd $WORKPATH/comps/retrievers/deployment/docker_compose

    docker compose -f compose.yaml down  ${service_name} --remove-orphans
    docker compose -f compose.yaml down ${service_name_mm} --remove-orphans

    cid=$(docker ps -aq --filter "name=tei-embedding-serving")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker
    build_docker_images

    start_service
    test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
    validate_microservice "$test_embedding"

    stop_docker

    start_multimodal_service
    test_embedding_multi=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(512)]; print(embedding)")
    validate_microservice "$test_embedding_multi" "$service_name_mm"
    validate_mm_microservice "$test_embedding_multi" "$service_name_mm"

    stop_docker

    echo y | docker system prune

}

main
