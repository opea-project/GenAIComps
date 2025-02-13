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
service_name="retriever-elasticsearch"

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
    export ELASTICSEARCH_PORT1=11608   # 11600-11699
    export ELASTICSEARCH_PORT2=11609
    export RETRIEVER_PORT=11610
    export HF_TOKEN=${HF_TOKEN}
    export ES_CONNECTION_STRING="http://${host_ip}:${ELASTICSEARCH_PORT1}"
    export INDEX_NAME="test-elasticsearch"
    export LOGFLAG=True

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 15s

    cd $WORKPATH
    bash ./tests/utils/wait-for-it.sh ${host_ip}:$RETRIEVER_PORT -s -t 100 -- echo "Retriever up"
    RETRIEVER_UP=$?
    if [ ${RETRIEVER_UP} -ne 0 ]; then
        echo "Could not start Retriever."
        return 1
    fi

    sleep 5s
    bash ./tests/utils/wait-for-it.sh ${host_ip}:$RETRIEVER_PORT -s -t 1 -- echo "Retriever still up"
    RETRIEVER_UP=$?
    if [ ${RETRIEVER_UP} -ne 0 ]; then
        echo "Retriever crashed."
        return 1
    fi
}

function validate_microservice() {
    test_embedding=$(python3 -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")


    result=$(http_proxy=''
    curl http://${host_ip}:$RETRIEVER_PORT/v1/retrieval \
        -X POST \
        -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" \
        -H 'Content-Type: application/json')
    if [[ $result == *"retrieved_docs"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs elasticsearch-vector-db >> ${LOG_PATH}/vectorstore.log
        docker logs ${service_name} >> ${LOG_PATH}/retriever-elasticsearch.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down  ${service_name} --remove-orphans
    cid=$(docker ps -aq --filter "name=elasticsearch-vector-db")
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
