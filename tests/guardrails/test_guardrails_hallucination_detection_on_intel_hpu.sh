#!/usr/bin/env bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork/
    VLLM_VER=$(git describe --tags "$(git rev-list --tags --max-count=1)")
    echo "Check out vLLM tag ${VLLM_VER}"
    git checkout ${VLLM_VER} &> /dev/null
    docker build --no-cache --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile.hpu -t opea/vllm-gaudi:comps --shm-size=128g .
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi built fail"
        exit 1
    else
        echo "opea/vllm-gaudi built successful"
    fi

    cd $WORKPATH
    docker build --no-cache -t opea/hallucination-detection:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/hallucination_detection/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/hallucination-detection built fail"
        exit 1
    else
        echo "opea/hallucination-detection built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export host_ip=$(hostname -I | awk '{print $1}')
    export LLM_MODEL_ID="PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct"
    export LLM_ENDPOINT_PORT=12210
    export vLLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export HALLUCINATION_DETECTION_PORT=11305
    export VLLM_SKIP_WARMUP=true
    export TAG=comps
    service_name="vllm-gaudi-server hallucination-detection-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    if [ $? -ne 0 ]; then
        echo "Microservice failed to start!"
        for service in $service_name; do
            echo "Logs for $service..."
            docker logs $service
        done
        exit 1
    fi
    echo "Microservice started"
    sleep 1m
}

function validate_microservice() {
    echo "Validate microservice started"
    DATA='{"messages":[{"role": "user", "content": "Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: \"PASS\" is the answer is faithful to the DOCUMENT and \"FAIL\" if the answer is not faithful to the DOCUMENT. Show your reasoning.\n\n--\nQUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):\n{question}\n\n--\nDOCUMENT:\n{document}\n\n--\nANSWER:\n{answer}\n\n--\n\n Your output should be in JSON FORMAT with the keys \"REASONING\" and \"SCORE\":\n{{\"REASONING\": <your reasoning as bullet points>, \"SCORE\": <your final score>}}"}], "max_tokens":600,"model": "PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct" }'

    echo "test 1 - Case with Hallucination (Invalid or Inconsistent Output)"
    DOCUMENT="750 Seventh Avenue is a 615 ft (187m) tall Class-A office skyscraper in New York City. 101 Park Avenue is a 629 ft tall skyscraper in New York City, New York."
    QUESTION=" 750 7th Avenue and 101 Park Avenue, are located in which city?"
    ANSWER="750 7th Avenue and 101 Park Avenue are located in Albany, New York"

    DATA1=$(echo $DATA | sed "s/{question}/$QUESTION/g; s/{document}/$DOCUMENT/g; s/{answer}/$ANSWER/g")
    printf "$DATA1\n"

    result=$(curl localhost:11305/v1/hallucination_detection -X POST -d "$DATA1" -H 'Content-Type: application/json')
    if [[ $result == *"FAIL"* ]]; then
        echo "Result correct."
    else
        docker logs hallucination-detection-server
        exit 1
    fi

    echo "test 2 - Case without Hallucination (Valid Output)"
    DOCUMENT=".......An important part of CDC’s role during a public health emergency is to develop a test for the pathogen and equip state and local public health labs with testing capacity. CDC developed an rRT-PCR test to diagnose COVID-19. As of the evening of March 17, 89 state and local public health labs in 50 states......"
    QUESTION="What kind of test can diagnose COVID-19?"
    ANSWER=" rRT-PCR test"

    DATA2=$(echo $DATA | sed "s/{question}/$QUESTION/g; s/{document}/$DOCUMENT/g; s/{answer}/$ANSWER/g")
    printf "$DATA2\n"

    result=$(curl localhost:11305/v1/hallucination_detection -X POST -d "$DATA2" -H 'Content-Type: application/json')
    if [[ $result == *"PASS"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs hallucination-detection-server
        exit 1
    fi
    echo "Validate microservice completed"
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=hallucination-detection-server" --filter "name=vllm-gaudi-server")
    echo "Shutdown legacy containers "$cid
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo "cleanup container images and volumes"
    echo y | docker system prune 2>&1 > /dev/null

}

main
