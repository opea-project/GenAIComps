#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/llm_utils.sh

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"
export DATA_PATH=${model_cache}

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="docsum-vllm"

function build_docker_images() {
    cd $WORKPATH
    source $(git rev-parse --show-toplevel)/.github/env/_vllm_versions.sh
    git clone --depth 1 -b ${VLLM_VER} --single-branch https://github.com/vllm-project/vllm.git && cd vllm
    docker build --no-cache -f docker/Dockerfile.cpu -t ${REGISTRY:-opea}/vllm:${TAG:-latest} --shm-size=128g .
    if [ $? -ne 0 ]; then
        echo "opea/vllm built fail"
        exit 1
    else
        echo "opea/vllm built successful"
    fi

    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/llm-docsum:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/doc-summarization/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-docsum built fail"
        exit 1
    else
        echo "opea/llm-docsum built successful"
    fi
}

function start_service() {
    local offline=${1:-false}
    export host_ip=${host_ip}
    export LLM_ENDPOINT_PORT=12107  # 12100-12199
    export DOCSUM_PORT=10507 #10500-10599
    export HF_TOKEN=${HF_TOKEN}
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export LLM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
    export MAX_INPUT_TOKENS=2048
    export MAX_TOTAL_TOKENS=4096
    export VLLM_SKIP_WARMUP=true
    export LOGFLAG=True

    service_name="docsum-vllm"
    if [[ "$offline" == "true" ]]; then
        service_name="docsum-vllm-offline"
        export offline_no_proxy="${host_ip}"
        prepare_models ${DATA_PATH} ${LLM_MODEL_ID} gpt2
    fi
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_doc-summarization.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 30s
}

function validate_services() {
    local URL="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_NAME="$3"
    local DOCKER_NAME="$4"
    local INPUT_DATA="$5"

    local HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL")

    echo "==========================================="

    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."

        local CONTENT=$(curl -s -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/${SERVICE_NAME}.log)

        echo $CONTENT

        if echo "$CONTENT" | grep -q "$EXPECTED_RESULT"; then
            echo "[ $SERVICE_NAME ] Content is as expected."
        else
            echo "[ $SERVICE_NAME ] Content does not match the expected result"
            docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_NAME}.log
            exit 1
        fi
    else
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_NAME}.log
        exit 1
    fi
    sleep 1s
}

function validate_microservices() {
    URL="http://${host_ip}:$DOCSUM_PORT/v1/docsum"

    echo "Validate vllm..."
    validate_services \
        "${LLM_ENDPOINT}/v1/completions" \
        "text" \
        "vllm-server" \
        "vllm-server" \
        '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt": "What is Deep Learning?", "max_tokens": 32, "temperature": 0}'

    echo "Validate stream=True..."
    validate_services \
        "$URL" \
        'text' \
        "docsum-vllm" \
        "docsum-vllm" \
        '{"messages": "Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en"}'

    echo "Validate stream=False..."
    validate_services \
        "$URL" \
        'text' \
        "docsum-vllm" \
        "docsum-vllm" \
        '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "stream":false}'

    echo "Validate Chinese mode..."
    validate_services \
        "$URL" \
        'text' \
        "docsum-vllm" \
        "docsum-vllm" \
        '{"messages":"2024年9月26日，北京——今日，英特尔正式发布英特尔® 至强® 6性能核处理器（代号Granite Rapids），为AI、数据分析、科学计算等计算密集型业务提供卓越性能。", "max_tokens":32, "language":"zh", "stream":false}'

    echo "Validate truncate mode..."
    validate_services \
        "$URL" \
        'text' \
        "docsum-vllm" \
        "docsum-vllm" \
        '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "summary_type": "truncate", "chunk_size": 2000}'

    echo "Validate map_reduce mode..."
    validate_services \
        "$URL" \
        'text' \
        "docsum-vllm" \
        "docsum-vllm" \
        '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "summary_type": "map_reduce", "chunk_size": 2000, "stream":false, "timeout":200}'

    echo "Validate refine mode..."
    validate_services \
        "$URL" \
        'text' \
        "docsum-vllm" \
        "docsum-vllm" \
        '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "summary_type": "refine", "chunk_size": 2000, "timeout":200}'
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_doc-summarization.yaml down --remove-orphans
}

function main() {

    stop_docker

    build_docker_images

    trap stop_docker EXIT

    echo "Test normal env ..."
    start_service
    validate_microservices
    stop_docker

    if [[ -n "${DATA_PATH}" ]]; then
        echo "Test air gapped env ..."
        start_service true
        validate_microservices
        stop_docker
    fi
    echo y | docker system prune

}

main
