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
export host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="vllm-openvino"

function build_container() {
    cd $WORKPATH
    git clone https://github.com/vllm-project/vllm.git vllm-openvino
    cd ./vllm-openvino/ && git checkout v0.6.1 # something wrong with main branch image build

    docker build --no-cache -t ${REGISTRY:-opea}/vllm-openvino:${TAG:-latest} \
      -f Dockerfile.openvino \
      . \
      --build-arg https_proxy=$https_proxy \
      --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "vllm-openvino built fail"
        exit 1
    else
        echo "vllm-openvino built successful"
    fi
    cd $WORKPATH
    rm -rf vllm-openvino
}

# Function to start Docker container
start_container() {
    export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
    export LLM_ENDPOINT_PORT=12205
    export HF_CACHE_DIR=${model_cache:-./data}

    cd $WORKPATH/comps/third_parties/vllm/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    # check whether service is fully ready
    n=0
    until [[ "$n" -ge 300 ]]; do
        docker logs $service_name > /tmp/$service_name.log 2>&1
        n=$((n+1))
        if grep -q "Uvicorn running on" /tmp/$service_name.log; then
            break
        fi
        sleep 3s
    done

}

# Function to test API endpoint
function test_api_endpoint {
    local endpoint="$1"
    local expected_status="$2"

    # Make the HTTP request
    if test "$1" = "v1/completions"
    then
        local response=$(curl "http://$host_ip:$LLM_ENDPOINT_PORT/$endpoint" \
          -H "Content-Type: application/json" \
          -d '{
                "model": "Intel/neural-chat-7b-v3-3",
                "prompt": "What is the key advantage of Openvino framework",
                "max_tokens": 300,
                "temperature": 0.7
              }' \
          --write-out '%{http_code}' \
          --silent \
          --output /dev/null)
    else
        local response=$(curl "http://$host_ip:$LLM_ENDPOINT_PORT/$endpoint" \
          --write-out '%{http_code}' \
          --silent \
          --output /dev/null)
    fi

    # Assert the response status code
    if [[ "$response" -eq "$expected_status" ]]; then
        echo "PASS: $endpoint returned expected status code: $expected_status"
    else
        echo "FAIL: $endpoint returned unexpected status code: $response (expected: $expected_status)"
        docker logs $service_name
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_faq-generation.yaml down --remove-orphans
}

# Main function
main() {

    build_container
    start_container

    # Sleep to allow the container to start up fully
    sleep 10
    # Test the /v1/models API
    test_api_endpoint "v1/models" 200

    # Test the /v1/completions API
    test_api_endpoint "v1/completions" 200

    stop_docker
}

# Call main function
main
