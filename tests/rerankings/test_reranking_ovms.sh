#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
service_name="ovms-reranking-server ovms-reranking-serving"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache \
          -t opea/reranking:comps \
          --build-arg https_proxy=$https_proxy \
          --build-arg http_proxy=$http_proxy \
          -f comps/rerankings/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/reranking built fail"
        exit 1
    else
        echo "opea/reranking built successful"
    fi
}

function get_model() {
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod o+rw /cache/hub || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod o+rw /cache/hub/.locks || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod -R o+rw /cache/hub/.locks/models--BAAI--bge-reranker-base || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod -R o+rw /cache/hub/models--BAAI--bge-reranker-base || true
    pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models
    python export_model.py rerank --source_model BAAI/bge-reranker-base --weight-format int8 --config_file_path models/config_reranking.json --model_repository_path models --target_device CPU
    chmod -R 755 models
}

function start_service() {
    export MODEL_ID="BAAI/bge-reranker-base"
    export OVMS_RERANKER_PORT=12004
    export RERANK_PORT=10702
    export MODELS_REPOSITORY=${PWD}/models
    export OVMS_RERANKING_ENDPOINT="http://${host_ip}:${OVMS_RERANKER_PORT}"
    export TAG=comps
    export host_ip=${host_ip}

    cd $WORKPATH/comps/rerankings/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d
    sleep 10
}

function validate_microservice() {
    ovms_service_port=10702
    local CONTENT=$(curl http://${host_ip}:${ovms_service_port}/v1/reranking \
        -X POST \
        -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
        -H 'Content-Type: application/json')

    if echo "$CONTENT" | grep -q "documents"; then
        echo "Content is as expected."
    else
        echo "Content does not match the expected result: $CONTENT"
        docker logs ovms-reranking-serving
        docker logs reranking-ovms-server
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/rerankings/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
    cd $WORKPATH
}

function main() {

    stop_docker

    build_docker_images
    get_model
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
