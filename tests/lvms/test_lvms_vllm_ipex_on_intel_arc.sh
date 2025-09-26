#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"

# Pick GPU type on intel-arc runner
# 0: B60, 1: A770
export ZE_AFFINITY_MASK=0

export HF_TOKEN=$HF_TOKEN
service_name="lvm-vllm-ipex-service"

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
export LLM_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
export LOAD_QUANTIZATION=fp8
export VLLM_PORT=41091

function start_service() {

    cd $WORKPATH/comps/lvms/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log
    sleep 30s

    # Check if service is ready
    # May include model downloading time
    n=0
    until [[ "$n" -ge 100 ]]; do
        echo "Waiting for the service:${service_name} to be ready"
        LAST_LINES=$(docker compose -f compose.yaml logs --tail=3 2>/dev/null)
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
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "Describe the image."
            },
            {
                "type": "image_url",
                "image_url": {
                "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
                }
            }
            ]
        }
        ],
        "max_tokens": 512
    }')
    echo result: ${result}

    if [[ $result == *"bear"* ]]; then
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
