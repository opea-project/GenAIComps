#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    docker build -t opea/dataprep-on-ray-redis:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/redis/docker/Dockerfile_ray .
}

function start_service() {
    # redis endpoint
    docker run -d --name="test-dataprep-redis-server" --runtime=runc -p 16379:16379 -p 18001:18001 redis/redis-stack:7.2.0-v9

    # dataprep-redis-server endpoint
    export REDIS_URL="redis://${ip_address}:16379"
    export INDEX_NAME="rag-redis"
    unset http_proxy
    docker run -d --name="test-dataprep-redis-endpoint" --runtime=runc -p 6007:6007 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e REDIS_URL=$REDIS_URL -e INDEX_NAME=$INDEX_NAME -e TEI_ENDPOINT=$TEI_ENDPOINT opea/dataprep-on-ray-redis:latest

    sleep 5
}

function validate_microservice() {
    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    python -c "$(cat << 'EOF'
import requests
import json
proxies = {'http':""}
url = 'http://localhost:6007/v1/dataprep'
urls = ["https://towardsdatascience.com/no-gpu-no-party-fine-tune-bert-for-sentiment-analysis-with-vertex-ai-custom-jobs-d8fc410e908b?source=rss----7f60cf5620c9---4"] * 20
payload = {"link_list": json.dumps(urls)}

resp = requests.post(url=url, data=payload, proxies=proxies) 
resp.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes
print("Request successful!")
EOF
)"
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-dataprep-redis*")
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
