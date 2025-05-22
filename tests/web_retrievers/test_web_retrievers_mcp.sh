#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export TAG=comps
export WEB_RETRIEVER_PORT=11902
export TEI_PORT=11903
export DATA_PATH=${model_cache}
export ENABLE_MCP=True
cd $WORKPATH

function build_docker_images() {
    docker build --no-cache -t opea/web-retriever:$TAG --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/web_retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/web-retriever built fail"
        exit 1
    else
        echo "opea/web-retriever built successful"
    fi
}

function start_service() {
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export TEI_EMBEDDING_ENDPOINT=http://${ip_address}:${TEI_PORT}
    export host_ip=${ip_address}
    export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}

    docker compose -f comps/web_retrievers/deployment/docker_compose/compose.yaml up -d
    sleep 15s
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/utils/validate_svc_with_mcp.py $ip_address $WEB_RETRIEVER_PORT "web_retriever"
    if [ $? -ne 0 ]; then
        docker logs tei-embedding-server
        docker logs web-retriever-service
        exit 1
    fi

}

function stop_docker() {
    docker ps -a --filter "name=tei-embedding-server" --filter "name=web-retriever-service" --format "{{.Names}}" | xargs -r docker stop
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
