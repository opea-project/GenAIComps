#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT="11106"
export TAG="comps"

function build_docker_images() {
    cd $WORKPATH

    # build dataprep image for pinecone
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f $WORKPATH/comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {
    export PINECONE_API_KEY=$PINECONE_KEY
    export PINECONE_INDEX_NAME="test-index"
    export HUGGINGFACEHUB_API_TOKEN=$HF_TOKEN

    service_name="dataprep-pinecone"
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate_microservice() {
    URL="http://$ip_address:${DATAPREP_PORT}/v1/dataprep/ingest"
    echo 'The OPEA platform includes: Detailed framework of composable building blocks for state-of-the-art generative AI systems including LLMs, data stores, and prompt engines' > ./dataprep_file.txt
    result=$(curl --noproxy $ip_address --location --request POST \
      --form 'files=@./dataprep_file.txt' $URL)
    if [[ $result == *"200"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs dataprep-pinecone-server
        exit 1
    fi
    DELETE_URL="http://$ip_address:${DATAPREP_PORT}/v1/dataprep/delete"
    result=$(curl --noproxy $ip_address --location --request POST \
      -d '{"file_path": "all"}' -H 'Content-Type: application/json' $DELETE_URL)
    if [[ $result == *"true"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs dataprep-pinecone-server
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-pinecone-server*")
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
