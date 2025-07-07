#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export TAG=comps
export PORT=7900
export service_name="prompt-template"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/prompt-template:$TAG -f  comps/prompt_template/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/prompt-template built fail"
        exit 1
    else
        echo "opea/prompt-template built successful"
    fi
}

function start_service() {
    unset http_proxy
    cd $WORKPATH/comps/prompt_template/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d
    sleep 10s
}

function validate_microservice() {
    local PORT=7900
    local service_name="prompt-template"
    echo "🔍 Validating $service_name service on port $PORT..."

    echo "📦 Case 1: Default template..."
    result1=$(http_proxy="" curl -s -X POST http://localhost:$PORT/v1/prompt_template \
        -H "Content-Type: application/json" \
        -d '{
              "data": {
                        "user_prompt": "What is Deep Learning?",
                        "reranked_docs": [{"text":"Deep Learning is..."}]
                      },
              "conversation_history": [
                {"question": "Hello", "answer": "Hello as well"},
                {"question": "How are you?", "answer": "I am good, thank you!"},
                {"question": "Who are you?", "answer": "I am a robot"}
              ],
              "system_prompt_template": "",
              "user_prompt_template": ""
            }')

    chat_template1=$(echo "$result1" | jq -r '.chat_template')
    if [[ "$chat_template1" == *"You are a helpful, respectful, and honest assistant to help the user with questions."* ]]; then
        echo "✅ Case 1 passed."
    else
        echo "❌ Case 1 failed: missing default assistant description."
        echo "$chat_template1"
    fi

    echo "📦 Case 2: Custom prompt template..."
    SYSTEM_PROMPT="### Please refer to the search results obtained from the local knowledge base. But be careful to not incorporate information that you think is not relevant to the question. If you don't know the answer to a question, please don't share false information. ### Search results: {reranked_docs} \n"
    USER_PROMPT="### Question: {initial_query} \n### Answer:"

    result2=$(curl -s -X POST http://localhost:$PORT/v1/prompt_template \
      -H "Content-Type: application/json" \
      -d "{
            \"data\": {
                      \"initial_query\": \"What is Deep Learning?\",
                      \"reranked_docs\": [{\"text\":\"Deep Learning is...\"}]
                    },
            \"system_prompt_template\": \"${SYSTEM_PROMPT}\",
            \"user_prompt_template\": \"${USER_PROMPT}\"
          }")

    chat_template2=$(echo "$result2" | jq -r '.chat_template')
    if [[ "$chat_template2" == *"refer to the search results obtained from the local knowledge base"* ]]; then
        echo "✅ Case 2 passed."
    else
        echo "❌ Case 2 failed: missing expected custom prompt content."
        echo "$chat_template2"
    fi

    echo "📦 Case 3: Translation template..."
    SYSTEM_PROMPT="### You are a helpful, respectful, and honest assistant to help the user with translations. Translate this from {source_lang} to {target_lang}.\n"
    USER_PROMPT="### Question: {initial_query} \n### Answer:"

    result3=$(curl -s -X POST http://localhost:$PORT/v1/prompt_template \
      -H "Content-Type: application/json" \
      -d "{
            \"data\": {
                        \"initial_query\":\"什么是深度学习？\",
                        \"source_lang\": \"chinese\",
                        \"target_lang\": \"english\"
                    },
            \"system_prompt_template\": \"${SYSTEM_PROMPT}\",
            \"user_prompt_template\": \"${USER_PROMPT}\"
          }")

    chat_template3=$(echo "$result3" | jq -r '.chat_template')
    if [[ "$chat_template3" == *"Translate this from chinese to english"* ]]; then
        echo "✅ Case 3 passed."
    else
        echo "❌ Case 3 failed: translation instruction missing."
        echo "$chat_template3"
    fi
}


function stop_docker() {
    cd $WORKPATH/comps/prompt_template/deployment/docker_compose
    docker compose -f compose.yaml down --remove-orphans
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
