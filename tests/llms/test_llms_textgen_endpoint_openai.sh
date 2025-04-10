#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# tests/llms/test_llms_textgen_remote_openai.sh
# SPDX-License-Identifier: Apache-2.0

# This script tests the textgen connection with an openai endpoint based on a vLLM openAI endpoint.
# The vLLM server is started in a docker container, and the textgen service is started in another container.
# The textgen service is configured to connect to the vLLM server using a test key.
# The test sends a request to the textgen service and validates the response.

set -e # Exit on error

# --- Check for jq ---
if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Please install it (e.g., apt install jq)."
    exit 1
fi

# --- Setup ---
WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
VLLM_MODEL=""Qwen/Qwen2.5-0.5B-Instruct""
LLM_ENDPOINT_PORT=8000


function build_vllm_image() {
    # This image is used to test textgen-service-endpoint-openai
    rm -rf $WORKPATH/vllm  # Remove existing vllm directory if it exists
    cd $WORKPATH

    # Pull the last tagged version of vLLM.
    git clone https://github.com/vllm-project/vllm.git && cd vllm
    VLLM_VER="$(git describe --tags "$(git rev-list --tags --max-count=1)" )"
    echo "Checked out vLLM tag ${VLLM_VER}"
    git checkout ${VLLM_VER} &> /dev/null

    docker build --no-cache -f docker/Dockerfile.cpu -t opea/vllm-cpu:test .
    cd $WORKPATH
}

function start_vllm() {
    export HF_TOKEN=${HF_TOKEN} # Remember to set HF_TOKEN before invoking this test!
    export VLLM_API_KEY=testkey # This is the VLLM environment variable to set keys.
    export host_ip=$(hostname -I | awk '{print $1}')

    docker run --rm -d \
        -p ${LLM_ENDPOINT_PORT}:8000 \
        -e VLLM_API_KEY=${VLLM_API_KEY} \
        --name vllm-server \
        opea/vllm-cpu:test \
        --model ${VLLM_MODEL} \
        --port 8000 \
        --host 0.0.0.0
    sleep 30
}

function start_textgen() {
    # Testing if the textgen can connect to a vllm endpoint, with a testkey.
    export OPENAI_API_KEY=testkey
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}" # Point to vLLM
    export LLM_MODEL_ID=$VLLM_MODEL  # Must match vLLM
    export service_name="textgen-service-endpoint-openai"
    export LOGFLAG=True

    # textgen-service-endpoint-openai extends the textgen service. This test uses the image: opea/llm-textgen:latest
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d
    sleep 20
}

function validate_service() {
    local response=$(curl -s -X POST http://${host_ip}:9000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer testkey" \
        -d '{
            "model": "'${VLLM_MODEL}'",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        }')

    echo "Raw response: $response"

    local status_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://${host_ip}:9000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer testkey" \
        -d '{
            "model": "'${VLLM_MODEL}'",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, world!"
                }
            ]
        }')

    if [[ "$status_code" -ne 200 ]]; then
        echo "Error: HTTP status code is not 200. Received: $status_code"
        docker logs textgen-service-endpoint-openai || true
        docker logs vllm-server || true
        exit 1
    fi

   local generated_text=$(echo "$response" | jq -r '.choices[0].message.content')

    if [[ -z "$generated_text" ]]; then
        echo "Error: No generated text found in response."
        docker logs textgen-service-endpoint-openai || true
        docker logs vllm-server || true
        exit 1
    fi

    echo "Test passed. Generated text: $generated_text"
}


function stop_containers() {
    docker stop textgen-service-endpoint-openai || true
    docker stop vllm-server || true
}

# Assumes containers from other test runs are already cleared.
build_vllm_image
start_vllm
start_textgen
validate_service
stop_containers
docker system prune -a -f

echo "All tests completed."
