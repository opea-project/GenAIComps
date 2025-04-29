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
service_name="tgi-rocm-server"
docker_container_name="tgi-rocm-server"

function build_container() {
    cd $WORKPATH/comps/third_parties/tgi/src
    docker build --no-cache -t ${REGISTRY:-opea}/tgi-rocm:${TAG:-latest} \
      -f Dockerfile.amd_gpu \
      . \
      --build-arg https_proxy=$https_proxy \
      --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "tgi-rocm built fail"
        exit 1
    else
        echo "tgi-rocm built successful"
    fi
}

# Function to start Docker container
start_container() {
    export HF_CACHE_DIR=${model_cache:-./data}
    export LLM_ENDPOINT_PORT=8008
    export host_ip=${host_ip}
    export HF_TOKEN=${HF_TOKEN}
    export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
    export MAX_INPUT_TOKENS=1024
    export MAX_TOTAL_TOKENS=2048

    cd $WORKPATH/comps/third_parties/tgi/deployment/docker_compose
    sudo chown -R 777 ${HF_CACHE_DIR}
    sudo mkdir ${HF_CACHE_DIR}/out && sudo chown -R 777 ${HF_CACHE_DIR}/out
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    # check whether service is fully ready
    n=0
    until [[ "$n" -ge 300 ]]; do
        docker logs ${docker_container_name} &> ${LOG_PATH}/${docker_container_name}.log 2>&1
        n=$((n+1))
        if grep -q "Connected" ${LOG_PATH}/${docker_container_name}.log; then
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
    DATA=
    if test "$1" = "generate"
    then
        local response=$(curl "http://${host_ip}:${LLM_ENDPOINT_PORT}/$endpoint" \
          -H "Content-Type: application/json" \
          -d '{"inputs":"What is a Deep Learning?","parameters":{"max_new_tokens":64,"do_sample": true}}' \
          --write-out '%{http_code}' \
          --silent \
          --output /dev/null)
    else
        local response=$(curl "http://${host_ip}:${LLM_ENDPOINT_PORT}/$endpoint" \
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
    cd $WORKPATH/../comps/third_parties/tgi/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
}

# Main function
main() {

    stop_docker

    build_container
    start_container

    # Sleep to allow the container to start up fully
    sleep 10

    # Test the /generate API
    test_api_endpoint "generate" 200

    stop_docker
}

# Call main function
main
