#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    dockerfile_name="comps/embeddings/src/$1"
    docker build --no-cache -t opea/embedding:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f "${dockerfile_name}" .
    if [ $? -ne 0 ]; then
        echo "opea/embedding built fail"
        exit 1
    else
        echo "opea/embedding built successful"
    fi
}

function start_service() {
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export TEI_EMBEDDER_PORT=12000
    export EMBEDDER_PORT=10200
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${TEI_EMBEDDER_PORT}"
    export TAG=comps
    export host_ip=${ip_address}
    service_name="tei-embedding-serving tei-embedding-server"
    cd $WORKPATH
    cd comps/embeddings/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_service() {
    local INPUT_DATA="$1"
    tei_service_port=10200
    result=$(http_proxy="" curl http://${ip_address}:$tei_service_port/v1/embeddings \
        -X POST \
        -d "$INPUT_DATA" \
        -H 'Content-Type: application/json')
    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs tei-embedding-serving
        docker logs tei-embedding-server
        exit 1
    fi
}

function validate_microservice() {
    ## Test OpenAI API, input single text
    validate_service \
        '{"input":"What is Deep Learning?"}'

    ## Test OpenAI API, input multiple texts with parameters
    validate_service \
        '{"input":["What is Deep Learning?","How are you?"], "dimensions":100}'
}

function validate_microservice_with_openai() {
    tei_service_port=10200
    pip install openai
    python3 ${WORKPATH}/tests/utils/validate_svc_with_openai.py $ip_address $tei_service_port "embedding"
    if [ $? -ne 0 ]; then
        docker logs tei-embedding-serving
        docker logs tei-embedding-server
        exit 1
    fi
}


function stop_service() {
    cd $WORKPATH/comps/embeddings/deployment/docker_compose/
    docker compose down || true
}

function main() {

    build_docker_images "Dockerfile"
    trap stop_service EXIT

    echo "Test normal env ..."
    start_service
    validate_microservice
    validate_microservice_with_openai
    stop_service

    echo "Test with openEuler OS ..."
    build_docker_images "Dockerfile.openEuler"
    start_service
    validate_microservice
    validate_microservice_with_openai
    stop_service

    docker system prune -f

}

main
