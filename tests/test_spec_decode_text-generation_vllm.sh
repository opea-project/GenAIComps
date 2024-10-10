#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    ## Build VLLM Ray docker
    cd $WORKPATH/comps/spec_decode/text-generation/vllm
    # Dockerfile default is GPU
    docker build \
        -f docker/Dockerfile \
        --no-cache -t opea/vllm-gpu:comps \
        --shm-size=128g .
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gpu built fail"
        exit 1
    else
        echo "opea/vllm-gpu built successful"
    fi

    ## Build OPEA microservice docker
    cd $WORKPATH
    docker build  \
        --no-cache -t opea/spec_decode-vllm:comps \
        -f comps/spec_decode/text-generation/vllm/docker/Dockerfile.microservice .
    if [ $? -ne 0 ]; then
        echo "opea/spec_decode-vllm built fail"
        exit 1
    else
        echo "opea/spec_decode-vllm built successful"
    fi
}

function start_service() {
    export SPEC_MODEL="facebook/opt-125m"
    export LLM_MODEL="facebook/opt-6.7b"
    port_number=5025
    docker run -d --rm \
        --gpus all \
        --name="test-comps-vllm-service" \
        -v $PWD/data:/data \
        -p $port_number:$port_number \
        -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
        --cap-add=sys_nice \
        --ipc=host \
        -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} \
        opea/vllm-gpu:comps \
        /bin/bash -c "export VLLM_CPU_KVCACHE_SPACE=40 && python3 -m vllm.entrypoints.openai.api_server --enforce-eager --model $LLM_MODEL --speculative_model $SPEC_MODEL --num_speculative_tokens 5 --gpu_memory_utilization 0.8 --use-v2-block-manager --seed 42  --tensor-parallel-size 1 --host 0.0.0.0 --port $port_number --block-size 128 --max-num-seqs 256 --max-seq_len-to-capture 2048"

    export vLLM_ENDPOINT="http://${ip_address}:${port_number}"
    docker run -d --rm \
        --name="test-comps-vllm-microservice" \
        -p 5030:5030 \
        --ipc=host \
        -e vLLM_ENDPOINT=$vLLM_ENDPOINT \
        -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN \
        -e LLM_MODEL=$LLM_MODEL \
        opea/spec_decode-vllm:comps

    n=0
    until [[ "$n" -ge 120 ]] || [[ $ready == true ]]; do
        docker logs test-comps-vllm-service > ${WORKPATH}/tests/test-comps-vllm-service.log
        n=$((n+1))
        if grep -q Connected ${WORKPATH}/tests/test-comps-vllm-service.log; then
            break
        fi
        sleep 5s
    done
    sleep 5s
}

function validate_microservice() {
    result=$(http_proxy="" curl http://${ip_address}:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "facebook/opt-6.7b",
        "prompt": "What is Deep Learning?",
        "max_tokens": 32,
        "temperature": 0
        }')
    if [[ $result == *"text"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        # docker logs test-comps-vllm-service
        # docker logs test-comps-vllm-microservice
        exit 1
    fi
    result=$(http_proxy="" curl http://${ip_address}:5003/v1/spec_decode/completions \
        -X POST \
        -d '{"query":"What is Deep Learning?","max_new_tokens":17,"top_p":0.95,"temperature":0.01,"streaming":false}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"text"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        # docker logs test-comps-vllm-service
        # docker logs test-comps-vllm-microservice
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-vllm*")
    if [[ ! -z "$cid" ]]; then docker rm $cid -f && sleep 1s; fi
}

function main() {

    # stop_docker

    # build_docker_images
    # start_service

    validate_microservice

    # stop_docker
    # echo y | docker system prune

}

main
