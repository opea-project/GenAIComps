#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/llm-vllm:comps -f comps/llms/text-generation/vllm/Dockerfile .
    cd $WORKPATH
    git clone https://github.com/vllm-project/vllm.git && cd ./vllm/ && docker build -f Dockerfile.cpu -t vllm:cpu --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
}

function start_service() {
    vllm_endpoint_port=5014
    export your_hf_llm_model="Intel/neural-chat-7b-v3-3"
    # Remember to set HF_TOKEN before invoking this test!
    export HF_TOKEN=${HF_TOKEN}
    docker run -d --name="test-comps-llm-vllm-endpoint" -p $vllm_endpoint_port:80 -v ./data:/data --shm-size 128g -e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$http_proxy -e HF_TOKEN=$HF_TOKEN vllm:cpu /bin/bash -c "cd / && export VLLM_CPU_KVCACHE_SPACE=40 && python3 -m vllm.entrypoints.openai.api_server --model $your_hf_llm_model --host 0.0.0.0 --port 80"
    export vLLM_LLM_ENDPOINT="http://${ip_address}:${vllm_endpoint_port}"

    llm_service_port=5015
    unset http_proxy
    docker run -d --name="test-comps-llm-vllm-server" -p ${llm_service_port}:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e vLLM_LLM_ENDPOINT=$vLLM_LLM_ENDPOINT -e HF_TOKEN=$HF_TOKEN -e LLM_MODEL_ID=$your_hf_llm_model opea/llm-vllm:comps

    # check whether vllm is fully ready
    n=0
    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
        docker logs test-comps-llm-vllm-endpoint > ${WORKPATH}/tests/test-comps-llm-vllm-endpoint.log
        n=$((n+1))
        if grep -q "Application startup complete" ${WORKPATH}/tests/test-comps-llm-vllm-endpoint.log; then
            break
        fi
        sleep 5s
    done
    sleep 5s
}

function validate_microservice() {
    llm_service_port=5011
    http_proxy="" curl http://${ip_address}:${llm_service_port}/v1/chat/completions \
        -X POST \
        -d '{"query":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json'
    docker logs test-comps-llm-vllm-endpoint
    docker logs test-comps-llm-vllm-server
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-llm-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
