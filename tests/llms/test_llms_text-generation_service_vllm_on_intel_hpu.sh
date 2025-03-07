#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="textgen-service-vllm-gaudi"

function build_docker_images() {
    cd $WORKPATH
    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork/
    VLLM_VER=$(git describe --tags "$(git rev-list --tags --max-count=1)")
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
    docker build --no-cache -t ${REGISTRY:-opea}/llm-textgen:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-textgen built fail"
        exit 1
    else
        echo "opea/llm-textgen built successful"
    fi
}

function start_service() {
    export LLM_ENDPOINT_PORT=12110  # 12100-12199
    export TEXTGEN_PORT=10510 #10500-10599
    export host_ip=${host_ip}
    export HF_TOKEN=${HF_TOKEN} # Remember to set HF_TOKEN before invoking this test!
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
    export VLLM_SKIP_WARMUP=true
    export LOGFLAG=True
    export DATA_PATH="/data2/cache"

    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

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

function validate_microservices() {
    URL="http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions"

    # vllm
    echo "Validate vllm..."
    validate_services \
        "${LLM_ENDPOINT}/v1/completions" \
        "text" \
        "vllm-gaudi-server" \
        "vllm-gaudi-server" \
        '{"model": "Intel/neural-chat-7b-v3-3", "prompt": "What is Deep Learning?", "max_tokens": 32, "temperature": 0}'

    # textgen
    echo "Validate textgen with string messages input..."
    validate_services \
        "$URL" \
        "text" \
        "textgen-service-vllm-gaudi" \
        "textgen-service-vllm-gaudi" \
        '{"model": "Intel/neural-chat-7b-v3-3", "messages": "What is Deep Learning?", "max_tokens":17, "stream":false}'

    echo "Validate textgen with dict messages input..."
    validate_services \
        "$URL" \
        "content" \
        "textgen-service-vllm-gaudi" \
        "textgen-service-vllm-gaudi" \
        '{"model": "Intel/neural-chat-7b-v3-3", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17, "stream":false}'
}

function validate_microservice_with_openai() {
    python3 ${WORKPATH}/tests/utils/validate_svc_with_openai.py "$host_ip" "$TEXTGEN_PORT" "llm"
    if [ $? -ne 0 ]; then
        docker logs vllm-gaudi-server >> ${LOG_PATH}/llm--gaudi.log
        docker logs textgen-service-vllm-gaudi >> ${LOG_PATH}/llm-server.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml down ${service_name} --remove-orphans
}

function main() {

    stop_docker

    build_docker_images
    pip install --no-cache-dir openai pydantic
    start_service

    validate_microservices
    validate_microservice_with_openai

    stop_docker
    echo y | docker system prune

}

main
