#!/bin/bash
# Copyright (c) 2024 Advanced Micro Devices, Inc.

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
WORKDIR=${WORKPATH}/../
export host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH"
service_name="vllm-rocm-server"
docker_container_name="vllm-server"

function build_container() {
    cd $WORKPATH/comps/third_parties/vllm/src
    docker build --no-cache -t ${REGISTRY:-opea}/vllm-rocm:${TAG:-latest} \
      -f Dockerfile.amd_gpu \
      . \
      --build-arg https_proxy=$https_proxy \
      --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "vllm-rocm built fail"
        exit 1
    else
        echo "vllm-rocm built successful"
    fi
}

# Function to start Docker container
start_container() {
    export VLLM_SERVICE_PORT=28011
    export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
    export HF_CACHE_DIR="./data"
    export VLLM_LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
    export TENSOR_PARALLEL_SIZE=1

    cd $WORKPATH/comps/third_parties/vllm/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    # check whether service is fully ready
    n=0
    until [[ "$n" -ge 300 ]]; do
        docker logs ${docker_container_name} > ${LOG_PATH}/${docker_container_name}.log 2>&1
        n=$((n+1))
        if grep -q "Application startup complete" ${LOG_PATH}/${docker_container_name}.log; then
            break
        fi
        sleep 10s
    done

}

# Function to test API endpoint
function test_api_endpoint {
    local endpoint="$1"
    local expected_status="$2"

    # Make the HTTP request
    if test "$1" = "v1/completions"
    then
        local response=$(curl "http://${host_ip}:${VLLM_SERVICE_PORT}/$endpoint" \
          -H "Content-Type: application/json" \
          -d '{
                "model": "Intel/neural-chat-7b-v3-3",
                "prompt": "What is deep learning?",
                "max_tokens": 300,
                "temperature": 0.7
              }' \
          --write-out '%{http_code}' \
          --silent \
          --output /dev/null)
    else
        local response=$(curl "http://${host_ip}:${VLLM_SERVICE_PORT}/$endpoint" \
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
    cd $WORKPATH/../comps/third_parties/vllm/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
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
