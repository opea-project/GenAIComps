#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
export DATA_PATH=${model_cache}

function build_docker_images() {
    echo "Start building docker images for microservice"
    cd $WORKPATH
    docker pull ghcr.io/huggingface/tgi-gaudi:2.3.1
    docker build --no-cache -t opea/guardrails:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/guardrails/src/guardrails/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/guardrails built fail"
        exit 1
    else
        echo "opea/guardrails built successful"
    fi
}

function start_service() {
    echo "Starting microservice"
    export host_ip=${ip_address}
    export LLM_MODEL_ID="meta-llama/Meta-Llama-Guard-2-8B"
    export LLM_ENDPOINT_PORT=12110
    export SAFETY_GUARD_MODEL_ID="meta-llama/Meta-Llama-Guard-2-8B"
    export SAFETY_GUARD_ENDPOINT=http://${ip_address}:${LLM_ENDPOINT_PORT}
    export GUARDRAILS_PORT=11303
    export TAG=comps
    service_name="tgi-gaudi-server llamaguard-guardrails-server"
    cd $WORKPATH
    cd comps/guardrails/deployment/docker_compose/
    docker compose up ${service_name} -d
    echo "Microservice started"
    sleep 15
}

function validate_microservice() {
    echo "Validate microservice started"
    echo "test 1 - violated policies"
    result=$(http_proxy= curl http://localhost:11303/v1/guardrails  -X POST   -d '{"text":"How do you buy a tiger in the US?","parameters":{"max_new_tokens":32}}'   -H 'Content-Type: application/json')
    if [[ $result == *"Violated"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs tgi-gaudi-server
        docker logs llamaguard-guardrails-server
        exit 1
    fi
    echo "test 2 - safe"
    result=$(http_proxy= curl http://localhost:11303/v1/guardrails  -X POST   -d '{"text":"How do you buy a car in the US?","parameters":{"max_new_tokens":32}}'   -H 'Content-Type: application/json')
    if [[ $result == *"car"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong."
        docker logs tgi-gaudi-server
        docker logs llamaguard-guardrails-server
        exit 1
    fi
    echo "test 3 - input guard"
    result=$(http_proxy= curl http://localhost:11303/v1/guardrails  -X POST  -d '{
            "query": "What are virus and backdoor?",
            "max_new_tokens": 17,
            "top_k": 10,
            "top_p": 0.95,
            "typical_p": 0.95,
            "temperature": 0.01,
            "repetition_penalty": 1.03,
            "stream": false,
            "input_guardrail_params": {
                "ban_substrings": {
                "enabled": true,
                "substrings": ["backdoor", "malware", "virus"],
                "case_sensitive": false
                }
            }
            }' \
        -H 'Content-Type: application/json'
       )
    if [[ "$result" == *"cannot assist you with your prompt"* ]]; then
        echo "✅ Input guard triggered as expected!"
    else
        echo "❌ Input guard failed to trigger!"
    fi

    echo "test 4 - output guard"
    result=$(http_proxy= curl http://localhost:11303/v1/guardrails  -X POST  -d '{"prompt":"What is Deep Learning?",
        "text": "To create a virus and backdoor you need to...",
        "output_guardrail_params":
            {"ban_substrings":
                {"enabled":true,
                "substrings":["backdoor","malware","virus"],
                "match_type":null,
                "case_sensitive":false,
                "redact":null,
                "contains_all":null}
            }
        }' \
        -H 'Content-Type: application/json'
       )

    echo "Output guard result:"
    echo "$result"

    if [[ "$result" != *"virus"* ]]; then
        echo "✅ Output guard triggered successfully (virus removed or replaced)!"
    else
        echo "❌ Output guard failed to trigger (virus still in output)!"
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=llamaguard-guardrails-server" --filter "name=tgi-gaudi-server")
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
