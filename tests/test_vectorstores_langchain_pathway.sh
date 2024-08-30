#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH

    cd comps/vectorstores/langchain/pathway

    docker build --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -t opea/vectorstore-pathway:comps .

    cd $WORKPATH

    if [ $? -ne 0 ]; then
        echo "opea/retriever-pathway built fail"
        exit 1
    else
        echo "opea/retriever-pathway built successful"
    fi
}

function start_service() {
    cd $WORKPATH

    # tei endpoint
    tei_endpoint=5008
    model="BAAI/bge-base-en-v1.5"
    docker run -d --name="test-comps-retriever-tei-endpoint" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p $tei_endpoint:80 -v ./data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 --model-id $model

    sleep 30s
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${tei_endpoint}"

    result=$(http_proxy=''
    curl $TEI_EMBEDDING_ENDPOINT     -X POST     -d '{"inputs":"Hey,"}'     -H 'Content-Type: application/json')

    echo "embed_result:"
    echo $result

    sleep 30s

    # pathway
    export PATHWAY_HOST="0.0.0.0"
    export PATHWAY_PORT=5432

    docker run -d --name="test-comps-vectorstore-pathway" -e PATHWAY_HOST=${PATHWAY_HOST} -e PATHWAY_PORT=${PATHWAY_PORT} -e TEI_EMBEDDING_ENDPOINT=${TEI_EMBEDDING_ENDPOINT} -e http_proxy=$http_proxy -e https_proxy=$https_proxy -v $WORKPATH/comps/vectorstores/langchain/pathway/README.md:/app/data/README.md -p ${PATHWAY_PORT}:${PATHWAY_PORT} --network="host" opea/vectorstore-pathway:comps

    sleep 45s

    export PATHWAY_HOST=$ip_address  # needed in order to reach to vector store

    sleep 10s
}

function validate_microservice() {
    export PATH="${HOME}/miniforge3/bin:$PATH"

    result=$(http_proxy=''
    curl http://${PATHWAY_HOST}:$PATHWAY_PORT/v1/retrieve \
        -X POST \
        -d "{\"query\":\"test\",\"k\":3}" \
        -H 'Content-Type: application/json')
    if [[ $result == *"Pathway"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-vectorstore-pathway >> ${LOG_PATH}/vectorstore-pathway.log
        docker logs test-comps-retriever-tei-endpoint >> ${LOG_PATH}/tei-endpoint.log
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps*")
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
