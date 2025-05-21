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
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="faqgen-vllm-gaudi"

function build_docker_images() {
    cd $WORKPATH
    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork/
    VLLM_VER=v0.6.6.post1+Gaudi-1.20.0
    echo "Check out vLLM tag ${VLLM_VER}"
    git checkout ${VLLM_VER} &> /dev/null
    docker build --no-cache -f Dockerfile.hpu -t ${REGISTRY:-opea}/vllm-gaudi:${TAG:-latest} --shm-size=128g .
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi built fail"
        exit 1
    else
        echo "opea/vllm-gaudi built successful"
    fi

    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/llm-faqgen:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/faq-generation/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-faqgen built fail"
        exit 1
    else
        echo "opea/llm-faqgen built successful"
    fi

}

function start_service() {
    export LLM_ENDPOINT_PORT=12102  # 12100-12199
    export FAQ_PORT=10502 #10500-10599
    export host_ip=${host_ip}
    export HF_TOKEN=${HF_TOKEN} # Remember to set HF_TOKEN before invoking this test!
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export LLM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
    export VLLM_SKIP_WARMUP=true
    export LOGFLAG=True
    export DATA_PATH=${model_cache:-./data}

    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_faq-generation.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

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

        if echo "$CONTENT" | grep -q "$EXPECTED_RESULT"; then
            echo "[ $SERVICE_NAME ] Content is as expected."
        else
            echo "[ $SERVICE_NAME ] Content does not match the expected result: $CONTENT"
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

function validate_backend_microservices() {
    # vllm
    validate_services \
        "${host_ip}:${LLM_ENDPOINT_PORT}/v1/completions" \
        "text" \
        "vllm-gaudi-server" \
        "vllm-gaudi-server" \
        '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "prompt": "What is Deep Learning?", "max_tokens": 32, "temperature": 0}'

    # faq
    validate_services \
        "${host_ip}:${FAQ_PORT}/v1/faqgen" \
        "text" \
        "faqgen-vllm-gaudi" \
        "faqgen-vllm-gaudi" \
        '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.","max_tokens": 32}'

    # faq, non-stream
    validate_services \
        "${host_ip}:${FAQ_PORT}/v1/faqgen" \
        "text" \
        "faqgen-vllm-gaudi" \
        "faqgen-vllm-gaudi" \
        '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.","max_tokens": 32, "stream":false}'
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_faq-generation.yaml down --remove-orphans
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_backend_microservices

    stop_docker
    echo y | docker system prune

}

main
