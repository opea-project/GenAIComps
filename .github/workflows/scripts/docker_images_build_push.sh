#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKSPACE=$(dirname "$PWD")
IMAGE_REPO=${IMAGE_REPO:-$OPEA_IMAGE_REPO}
IMAGE_TAG=${IMAGE_TAG:-latest}

function docker_build() {
    # docker_build <IMAGE_NAME>
    IMAGE_NAME=$1
    micro_service=$2
    dockerfile_path=${WORKSPACE}/comps/${micro_service}
    if [ -f "$dockerfile_path/Dockerfile" ]; then
        DOCKERFILE_PATH="$dockerfile_path/Dockerfile"
    elif [ -f "$dockerfile_path/docker/Dockerfile" ]; then
        DOCKERFILE_PATH="$dockerfile_path/docker/Dockerfile"
    else
        echo "Dockerfile not found"
        exit 1
    fi
    echo "Building ${IMAGE_REPO}/${IMAGE_NAME}:$IMAGE_TAG using Dockerfile $DOCKERFILE_PATH"

    docker build --no-cache -t ${IMAGE_REPO}/${IMAGE_NAME}:$IMAGE_TAG -f $DOCKERFILE_PATH .
    docker push ${IMAGE_REPO}/${IMAGE_NAME}:$IMAGE_TAG
    docker rmi ${IMAGE_REPO}/${IMAGE_NAME}:$IMAGE_TAG
}

# $1 is like micro_service_list
micro_service_list=$1
echo "micro_service_list: ${micro_service_list}"

for micro_service in ${micro_service_list}; do

case ${micro_service} in
    "asr"|"tts")
        IMAGE_NAME="opea/${micro_service}"
        ;;
    "embeddings/langchain")
        IMAGE_NAME="opea/embedding-tei"
        ;;
    "retrievers/langchain")
        IMAGE_NAME="opea/retriever-redis"
        ;;
    *)
        echo "Not supported yet"
        ;;
esac

docker_build "${IMAGE_NAME}" "${micro_service}"

done
