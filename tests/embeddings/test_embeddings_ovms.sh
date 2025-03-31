#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/embedding:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/embeddings/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/embedding built fail"
        exit 1
    else
        echo "opea/embedding built successful"
    fi
}

function get_model() {
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod o+rw /cache/hub || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod o+rw /cache/hub/.locks || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod -R o+rw /cache/hub/.locks/models--thenlper--gte-small || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod -R o+rw /cache/hub/models--thenlper--gte-small || true
    pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models
    python export_model.py embeddings --source_model thenlper/gte-small --weight-format int8 --config_file_path models/config_embeddings.json --model_repository_path models --target_device CPU
    chmod -R 755 models
}

function start_service() {
    export OVMS_EMBEDDER_PORT=10206
    export EMBEDDER_PORT=10205
    export MODELS_REPOSITORY=${PWD}/models
    export MODEL_ID="thenlper/gte-small"
    export OVMS_EMBEDDING_ENDPOINT="http://${ip_address}:${OVMS_EMBEDDER_PORT}"
    export TAG=comps
    export host_ip=${ip_address}
    service_name="ovms-embedding-serving ovms-embedding-server"
    cd $WORKPATH
    cd comps/embeddings/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 15
}

function validate_service() {
    local INPUT_DATA="$1"
    ovms_service_port=10205
    result=$(http_proxy="" curl http://${ip_address}:$ovms_service_port/v1/embeddings \
        -X POST \
        -d "$INPUT_DATA" \
        -H 'Content-Type: application/json')
    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs ovms-embedding-serving
        docker logs ovms-embedding-server
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
    ovms_service_port=10205
    pip install openai
    python3 ${WORKPATH}/tests/utils/validate_svc_with_openai.py $ip_address $ovms_service_port "embedding"
    if [ $? -ne 0 ]; then
        docker logs ovms-embedding-serving
        docker logs ovms-embedding-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=ovms-embedding-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    get_model
    start_service

    validate_microservice
    validate_microservice_with_openai

    stop_docker
    echo y | docker system prune

}

main
