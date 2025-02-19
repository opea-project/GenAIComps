#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
export host_ip=$(hostname -I | awk '{print $1}')
service_name="retriever-pathway"

function build_docker_images() {
    cd $WORKPATH

    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t ${REGISTRY:-opea}/vectorstore-pathway:${TAG:-latest}  -f comps/third_parties/pathway/src/Dockerfile .

    docker build --no-cache -t ${REGISTRY:-opea}/retriever:${TAG:-latest}  --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever built fail"
        exit 1
    else
        echo "opea/retriever built successful"
    fi
}

function start_service() {
    export TEI_EMBEDDER_PORT=11619
    export PATHWAY_PORT=11620
    export RETRIEVER_PORT=11621
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export PATHWAY_HOST_DB="0.0.0.0"
    export PATHWAY_VOLUME="$WORKPATH/comps/third_parties/pathway/src/README.md"
    export PATHWAY_HOST=$host_ip  # needed in order to reach to vector store

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 2m
}

function validate_microservice() {
    export PATH="${HOME}/miniforge3/bin:$PATH"

    test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

    result=$(http_proxy=''
    curl http://${host_ip}:$RETRIEVER_PORT/v1/retrieval \
        -X POST \
        -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" \
        -H 'Content-Type: application/json')
    if [[ $result == *"retrieved_docs"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs pathway-db >> ${LOG_PATH}/vectorstore-pathway.log
        docker logs tei-embedding-serving >> ${LOG_PATH}/tei-endpoint.log
        docker logs ${service_name} >> ${LOG_PATH}/retriever-pathway.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down  ${service_name} --remove-orphans
    cid=$(docker ps -aq --filter "name=pathway-db")
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
