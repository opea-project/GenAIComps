#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"

# Pick GPU type on intel-arc runner
# 0: B60, 1: A770
export ZE_AFFINITY_MASK=0

export HF_TOKEN=$HF_TOKEN
service_name="textgen-vllm-ipex-service"

export REGISTRY=intel
export TAG=1.0
echo "REGISTRY=${IMAGE_REPO}"
echo "TAG=${TAG}"

host_ip=$(hostname -I | awk '{print $1}')
export VIDEO_GROUP_ID=$(getent group video | awk -F: '{printf "%s\n", $3}')
export RENDER_GROUP_ID=$(getent group render | awk -F: '{printf "%s\n", $3}')

HF_HOME=${HF_HOME:=$HOME/.cache/huggingface}
export HF_HOME

export MAX_MODEL_LEN=20000
export LLM_MODEL_ID="Qwen/Qwen3-8B-AWQ"
export LOAD_QUANTIZATION=awq
export VLLM_PORT=41090

function start_service() {

    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log
    sleep 30s

    # Check if service is ready
    # May include model downloading time
    n=0
    until [[ "$n" -ge 100 ]]; do
        echo "Waiting for the service:${service_name} to be ready"
        LAST_LINES=$(docker compose -f compose_text-generation.yaml logs --tail=3 2>/dev/null)
        if echo "$LAST_LINES" | grep -q "Application startup complete"; then
            break
        fi
        sleep 6s
        n=$((n+1))
    done
    echo "The service:${service_name} is ready"
}

function validate_microservice() {
    URL="http://${host_ip}:${VLLM_PORT}/v1/chat/completions"

    result=$(curl ${URL} -XPOST -H "Content-Type: application/json" -d '{
        "model": "Qwen/Qwen3-8B-AWQ",
        "messages": [
        {
            "role": "user",
            "content": "What is Deep Learning?"
        }
        ],
        "max_tokens": 512
    }')
    echo result: ${result}

    if [[ $result == *"deep"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs ${service_name} >> ${LOG_PATH}/${service_name}.log
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=${service_name}" --format "{{.Names}}" | xargs -r docker stop
}

function main() {

    stop_docker

    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
