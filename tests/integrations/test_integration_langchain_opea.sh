#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"

function build_integration_package() {
    cd $WORKPATH/comps/integrations/langchain
    echo $(pwd)
    python3 -m venv /tmp/temp_env
    source /tmp/temp_env/bin/activate
    pip install --upgrade --force-reinstall poetry==1.8.4
    poetry install --with test
    if [ $? -ne 0 ]; then
        echo "Package installation fail"
        exit 1
    else
        echo "Package installation successful"
    fi
}

function start_service() {
    tei_endpoint=6006
    model="BAAI/bge-base-en-v1.5"
    unset http_proxy
    docker run -d --name="test-comps-integration-embedding-endpoint" -p $tei_endpoint:80 -v ./data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
    sleep 3m

    tgi_endpoint_port=9009
    export hf_llm_model="Intel/neural-chat-7b-v3-3"
    # Remember to set HF_TOKEN before invoking this test!
    export HF_TOKEN=${HF_TOKEN}
    docker run -d --name="test-comps-integration-llm-tgi-endpoint" -p $tgi_endpoint_port:80 -v ~/.cache/huggingface/hub:/data --shm-size 1g -e HF_TOKEN=${HF_TOKEN} ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id ${hf_llm_model} --max-input-tokens 1024 --max-total-tokens 2048
    # check whether tgi is fully ready
    n=0

    file=$LOG_PATH/test-comps-vllm-service.log

    if [ -f "$file" ] ; then
        rm "$file"
    fi

    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
        docker logs test-comps-integration-llm-tgi-endpoint >> $LOG_PATH/test-comps-vllm-service.log
        n=$((n+1))
        if grep -q Connected $LOG_PATH/test-comps-vllm-service.log; then
            break
        fi
        sleep 5s
    done
    sleep 5s

}

function validate_service() {
    tei_service_port=6006

    result=$(http_proxy="" curl http://${ip_address}:$tei_service_port/v1/embeddings \
        -X POST \
        -d '{"input":"What is Deep Learning?"}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"embedding"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-integration-embedding-endpoint
        exit 1
    fi

    llm_port=9009

    result=$(http_proxy="" curl http://${ip_address}:${llm_port}/v1/chat/completions \
        -X POST \
        -d '{"model": "Intel/neural-chat-7b-v3-3", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17, "stream":false}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"content"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs test-comps-llm-tgi-endpoint >> ${LOG_PATH}/llm-tgi.log
        exit 1
    fi

}

function validate_package(){
    cd "$WORKPATH/comps/integrations/langchain/"
    poetry run pytest --asyncio-mode=auto tests/
    if [ $? -ne 0 ]; then
        echo "Package Tests failed"
        exit 1
    else
        echo "Package Tests successful"
    fi

}


function remove_integration_package(){
    cd "$WORKPATH/comps/integrations/langchain/"
    deactivate
    rm -rf /tmp/temp_env
    if [ $? -ne 0 ]; then
        echo "Package removal fail"
        exit 1
    else
        echo "Package removal successful"
    fi

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-integration-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi

    ports=(6006 9009)
    for port in "${ports[@]}"; do
        docker ps --format "{{.ID}}" | while read -r container_id; do
            if docker port "$container_id" | grep ":$port"; then
                echo "Stopping container ($container_id)"
                docker stop "$container_id"
                echo "Removing container ($container_id)"
                docker rm "$container_id"
            fi
        done
    done

}


function main() {

    stop_docker

    build_integration_package

    start_service

    validate_service

    validate_package

    remove_integration_package

    stop_docker

    echo y | docker system prune

}

main
