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
service_name="retriever-pgvector"

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
    export PGVECTOR_PORT=11617
    export RETRIEVER_PORT=11618
    export POSTGRES_USER=testuser
    export POSTGRES_PASSWORD=testpwd
    export POSTGRES_DB=vectordb
    export HF_TOKEN=${HF_TOKEN}
    export VOLUMES_PATH=$WORKPATH/comps/third_parties/pgvector/src/init.sql
    export PG_CONNECTION_STRING=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@$host_ip:$PGVECTOR_PORT/${POSTGRES_DB}
    export LOGFLAG=True

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 1m
}

function validate_microservice() {
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
        docker logs pgvector-db >> ${LOG_PATH}/vectorstore.log
        docker logs ${service_name} >> ${LOG_PATH}/retriever-pgvector.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
    cid=$(docker ps -aq --filter "name=pgvector-db")
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
