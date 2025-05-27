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
export WORKPATH=$(dirname "$PWD")
export host_ip=$(hostname -I | awk '{print $1}')
export http_proxy=""
export LOG_PATH="$WORKPATH/tests"
export VLLM_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
export LLM_ENDPOINT_PORT="8000"
export OPENAI_API_KEY=testkey


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
    export VLLM_API_KEY=${OPENAI_API_KEY} # This is the VLLM environment variable to set keys.
    export host_ip=$(hostname -I | awk '{print $1}')

    # Block size must be 16 for CPU backend unless Intel Extension for PyTorch (IPEX) is installed
    BLOCK_SIZE=16   # If IPEX is installed can use 128.

    # Environment variables for vLLM CPU configuration:
    # - VLLM_USE_CPU=1: Explicitly force CPU backend usage
    # - VLLM_CPU_OMP_THREADS_BIND=all: Configure OpenMP to use all available CPU threads
    # - VLLM_CPU_KVCACHE_SPACE=4: Allocate 4GB of CPU memory for KV cache
    # - VLLM_MLA_DISABLE=1: Disable MLA (Multi-head Linear Attention) optimizations which aren't supported on CPU
    docker run --rm -d \
        -p ${LLM_ENDPOINT_PORT}:8000 \
        -e VLLM_API_KEY=${OPENAI_API_KEY} \
        -e VLLM_USE_CPU=1 \
        -e VLLM_CPU_OMP_THREADS_BIND=all \
        -e VLLM_CPU_KVCACHE_SPACE=4 \
        -e VLLM_MLA_DISABLE=1 \
        --name vllm-server \
        --network bridge \
        opea/vllm-cpu:test \
        --model ${VLLM_MODEL} \
        --port 8000 \
        --host 0.0.0.0 \
        --block-size ${BLOCK_SIZE}

    echo "Waiting for vLLM server to initialize..."
    # Wait for server to be ready by checking logs
    while ! docker logs vllm-server 2>&1 | grep -q "Application startup complete"; do
        echo "  Still waiting for vLLM server..."
        sleep 10
    done
    echo "vLLM server is ready!"

    # Verify server is responding
    curl -v http://localhost:${LLM_ENDPOINT_PORT}/health 2>&1 || echo "Warning: vLLM health check failed"
}

function start_textgen() {
    # Testing if the textgen can connect to a vllm endpoint, with a testkey.
    export OPENAI_API_KEY=testkey
    # Use host.docker.internal to access the host machine from within the container
    export LLM_ENDPOINT="http://${host_ip}:8000"
    export LLM_MODEL_ID=$VLLM_MODEL  # Must match vLLM
    export service_name="textgen-service-endpoint-openai"
    export LOGFLAG=True

    echo "Starting textgen service connecting to vllm at: ${LLM_ENDPOINT}"

    # textgen-service-endpoint-openai extends the textgen service. This test uses the image: opea/llm-textgen:latest
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml up ${service_name} -d

    echo "Waiting for textgen-service-endpoint-openai to initialize..."
    while ! docker logs textgen-service-endpoint-openai 2>&1 | grep -q "Application startup complete"; do
        echo "  Still waiting for textgen-service-endpoint-openai server..."
        sleep 5
    done
    echo "textgen-service-endpoint-openai is ready!"

    curl http://localhost:9000/health 2>&1 || echo "Warning: textgen health check failed"
}

function validate_chat_completions() {
    echo "Validating chat completions endpoint"
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

    echo "Raw chat response: $response"

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
        echo "Error: Chat completions HTTP status code is not 200. Received: $status_code"
        docker logs textgen-service-endpoint-openai || true
        docker logs vllm-server || true
        exit 1
    fi

    local generated_text=$(echo "$response" | jq -r '.choices[0].message.content')

    if [[ -z "$generated_text" ]]; then
        echo "Error: No generated text found in chat response."
        docker logs textgen-service-endpoint-openai || true
        docker logs vllm-server || true
        exit 1
    fi

    echo "Chat completions test passed. Generated text: $generated_text"
}


function stop_containers() {
    docker compose -f compose_text-generation.yaml down
    docker stop vllm-server || true # the --rm flag will ensure it is removed
}


# Assumes containers from other test runs are already cleared.
build_vllm_image
start_vllm
start_textgen
validate_chat_completions
stop_containers
docker system prune -a -f

echo "All tests completed."
