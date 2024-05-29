#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/microservices"
ip_address=$(hostname -I | awk '{print $1}')


function build_embedding_docker_image() {
    cd ${WORKPATH}
    cd ..
    docker build -t opea/embedding-tei:latest -f comps/embeddings/langchain/docker/Dockerfile .
    docker images
}

function start_embedding_service() {
    cd ${WORKPATH}
    cd ../comps/embeddings/langchain/docker

    export EMBEDDING_MODEL_ID="BAAI/bge-large-en-v1.5"
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:6066"

    # Start Docker Containers
    docker compose -f docker_compose_embedding.yaml up -d
    n=0
    until [[ "$n" -ge 200 ]]; do
        docker logs tei-xeon-server > ${LOG_PATH}/tei-xeon-server.log
        if grep -q Ready ${LOG_PATH}/tei-xeon-server.log; then
            break
        fi
        sleep 1s
        n=$((n+1))
    done
}

function validate_embedding_service() {
    # Check if the tei service is running correctly.
    curl ${ip_address}:6066/embed \
        -X POST \
        -d '{"inputs":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/curl_tei.log
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Microservice failed, please check the logs in artifacts!"
        docker logs tei-embedding-server >> ${LOG_PATH}/curl_tei.log
        exit 1
    fi
    sleep 1s

    # Check if the embedding service is running correctly.
    curl http://${ip_address}:6000/v1/embeddings \
        -X POST \
        -d '{"text":"hello"}' \
        -H 'Content-Type: application/json' > ${LOG_PATH}/curl_embeddings.log

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Embedding microservice failed, please check the logs in artifacts!"
        docker logs embedding-tei-server >> ${LOG_PATH}/curl_embeddings.log
        exit 1
    fi

    echo "Checking response results, make sure the output is reasonable. "
    local status=false
    if [[ -f $LOG_PATH/curl_embeddings.log ]] && \
    [[ $(grep -c "hello" $LOG_PATH/curl_embeddings.log) != 0 ]]; then
        status=true
    fi

    if [ $status == false ]; then
        echo "Response check failed, please check the logs in artifacts!"
        exit 1
    else
        echo "Response check succeed!"
    fi

}

function stop_docker() {
    cd $WORKPATH
    cd ../comps/embeddings/langchain/docker
    container_list=$(cat docker_compose_embedding.yaml | grep container_name | cut -d':' -f2)
    for container_name in $container_list; do
        cid=$(docker ps -aq --filter "name=$container_name")
        if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    done
    echo "All docker containers stopped and deleted"
}

function main() {

    stop_docker

    begin_time=$(date +%s)
    build_embedding_docker_image
    start_time=$(date +%s)
    start_embedding_service
    end_time=$(date +%s)
    minimal_duration=$((end_time-start_time))
    maximal_duration=$((end_time-begin_time))
    echo "Mega service start minimal duration is "$minimal_duration"s, maximal duration(including docker image build) is "$maximal_duration"s"

    validate_embedding_service

    stop_docker
    echo y | docker system prune

}

main
