#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH="$( cd "$( dirname "$0" )" && pwd )"

# Define variables
port=8123
HF_CACHE_DIR=$HOME/.cache/huggingface
DOCKER_IMAGE="vllm-openvino:comps"
CONTAINER_NAME="vllm-openvino-container"

function build_container() {
    cd $WORKPATH
    git clone https://github.com/vllm-project/vllm.git vllm-openvino
    cd ./vllm-openvino/
    docker build -t $DOCKER_IMAGE \
      -f Dockerfile.openvino \
      . \
      --build-arg https_proxy=$https_proxy \
      --build-arg http_proxy=$http_proxy
    cd $WORKPATH
    rm -rf vllm-openvino
}

# Function to start Docker container
start_container() {

    docker run -d --rm --name=$CONTAINER_NAME \
      -p $port:$port \
      --ipc=host \
      -e HTTPS_PROXY=$https_proxy \
      -e HTTP_PROXY=$https_proxy \
      -v $HF_CACHE_DIR:/root/.cache/huggingface \
      vllm-openvino:comps /bin/bash -c "\
        cd / && \
        export VLLM_CPU_KVCACHE_SPACE=50 && \
        python3 -m vllm.entrypoints.openai.api_server \
          --model \"Intel/neural-chat-7b-v3-3\" \
          --host 0.0.0.0 \
          --port $port"

    # check whether service is fully ready
    n=0
    until [[ "$n" -ge 300 ]]; do
        docker logs $CONTAINER_NAME > /tmp/$CONTAINER_NAME.log 2>&1
        n=$((n+1))
        if grep -q "Uvicorn running on" /tmp/$CONTAINER_NAME.log; then
            break
        fi
        sleep 3s
    done

}

# Cleanup Function
cleanup() {
    # Stop and remove Docker container and images
    cid=$(docker ps -aq --filter "name=$CONTAINER_NAME")
        if [[ ! -z "$cid" ]]; then docker stop $cid || docker rm $cid && sleep 1s; fi
    docker rmi -f $DOCKER_IMAGE
    rm /tmp/$CONTAINER_NAME.log
}

# Function to test API endpoint
function test_api_endpoint {
    local endpoint="$1"
    local expected_status="$2"

    # Make the HTTP request
    if test "$1" = "v1/completions"
    then
        local response=$(curl "http://localhost:$port/$endpoint" \
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
        local response=$(curl "http://localhost:$port/$endpoint" \
          --write-out '%{http_code}' \
          --silent \
          --output /dev/null)
    fi

    # Assert the response status code
    if [[ "$response" -eq "$expected_status" ]]; then
        echo "PASS: $endpoint returned expected status code: $expected_status"
    else
        echo "FAIL: $endpoint returned unexpected status code: $response (expected: $expected_status)"
    fi
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

    cleanup
}

# Call main function
main
