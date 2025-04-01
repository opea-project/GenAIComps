#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="textgen-service-ovms ovms-llm-serving"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/llm-textgen:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-textgen built fail"
        exit 1
    else
        echo "opea/llm-textgen built successful"
    fi
}


function get_model() {
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod o+rw /cache/hub || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod o+rw /cache/hub/.locks || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod -R o+rw /cache/hub/.locks/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0 || true
    docker run -v ${HOME}/.cache/huggingface:/cache ubuntu:22.04 chmod -R o+rw /cache/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0 || true
    pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
    mkdir -p models
    python export_model.py text_generation --source_model $1 --weight-format int8 --config_file_path models/config_llm.json --model_repository_path models --target_device CPU
    chmod -R 755 models
}

function start_service() {
    export OVMS_LLM_PORT=12115  # 12100-12199
    export TEXTGEN_PORT=10515 #10500-10599
    export host_ip=${host_ip}
    export LLM_ENDPOINT="http://${host_ip}:${OVMS_LLM_PORT}"
    export LLM_COMPONENT_NAME="OpeaTextGenOVMS"
    export MODEL_ID=$1
    export MODELS_REPOSITORY=${PWD}/models
    export LOGFLAG=True

    cd $WORKPATH
    cd comps/llms/deployment/docker_compose/
    docker compose -f compose_text-generation.yaml up ${service_name} -d
    sleep 5
}

function validate_microservice() {
    result=$(http_proxy="" curl http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
        -X POST \
        -d '{"messages": [{"role": "user", "content": "What is Deep Learning?"}]}' \
        -H 'Content-Type: application/json')
    if [[ $result == *"content"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs ovms-llm-serving
        docker logs textgen-service-ovms
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    docker compose -f compose_text-generation.yaml down ${service_name} --remove-orphans
    cd $WORKPATH
}

function main() {

    stop_docker
    build_docker_images

    llm_models=(
    TinyLlama/TinyLlama-1.1B-Chat-v1.0
    )
    for model in "${llm_models[@]}"; do
      get_model "${model}"
      start_service "${model}"
      validate_microservice
      stop_docker
    done

}

main
