#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/bedrock:latest \
        --build-arg https_proxy=$https_proxy \
        --build-arg http_proxy=$http_proxy \
        -f comps/llms/text-generation/bedrock/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/bedrock:latest built fail"
        exit 1
    else
        echo "opea/bedrock:latest built successful"
    fi
}

function start_service() {
    # Check for required AWS credentials
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        echo "AWS credentials not set in environment"
        exit 1
    fi

    docker run -d --name="bedrock-test" \
        -p 9009:9000 \
        --ipc=host \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
        opea/bedrock:latest

    # Give the service time to start
    sleep 10s
}

function validate_microservice() {
    bedrock_port=9009
    result=$(http_proxy="" curl http://${ip_address}:${bedrock_port}/v1/chat/completions \
        -X POST \
        -d '{"model": "us.anthropic.claude-3-haiku-20240307-v1:0", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17, "stream": "true"}' \
        -H 'Content-Type: application/json')

    if [[ $result == *"data: [DONE]"* ]]; then
        echo "Result correct."
        echo "$result" >> ${LOG_PATH}/bedrock.log
    else
        echo "Result wrong. Received was $result"
        docker logs bedrock-test >> ${LOG_PATH}/bedrock.log
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=bedrock-test")
    if [[ ! -z "$cid" ]]; then
        docker stop $cid && docker rm $cid && sleep 1s
    fi
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
