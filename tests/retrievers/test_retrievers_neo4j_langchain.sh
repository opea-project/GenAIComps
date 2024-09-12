#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/retriever-neo4j:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/neo4j/langchain/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever-neo4j built fail"
        exit 1
    else
        echo "opea/retriever-neo4j built successful"
    fi
}

function start_service() {
    # neo4j
    docker run -d -p 7474:7474 -p 7687:7687 -v $PWD/data:/data -v $PWD/plugins:/plugins --name neo4j-apoc -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS=\[\"apoc\"\] neo4j:latest
    sleep 60s

    # tei endpoint
    tei_endpoint=5434
    model="BAAI/bge-base-en-v1.5"
    docker run -d --name="test-comps-retriever-neo4j-tei-endpoint" -p $tei_endpoint:80 -v ./data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
    sleep 30s
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${tei_endpoint}"

    # Neo4J retriever
    export NEO4J_URI="bolt://${ip_address}:7687"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="password"
    # unset http_proxy
    docker run -d --name="test-comps-retriever-neo4j-server" -p ${retriever_port}:7000 --ipc=host -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e NEO4J_URI=$NEO4J_URI -e NEO4J_USERNAME=$NEO4J_USERNAME NEO4J_PASSWORD=$NEO4J_PASSWORD opea/retriever-neo4j:comps

    sleep 3m
}

function validate_microservice() {
    retriever_port=5435
    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    URL="http://${ip_address}:$retriever_port/v1/retrieval"

    test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")

    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content is as expected."
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs test-comps-retriever-redis-server >> ${LOG_PATH}/retriever.log
            docker logs test-comps-retriever-redis-tei-endpoint >> ${LOG_PATH}/tei.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs test-comps-retriever-redis-server >> ${LOG_PATH}/retriever.log
        docker logs test-comps-retriever-redis-tei-endpoint >> ${LOG_PATH}/tei.log
        exit 1
    fi
}

function stop_docker() {
    cid_retrievers=$(docker ps -aq --filter "name=test-comps-retriever-neo4j*")
    if [[ ! -z "$cid_retrievers" ]]; then
        docker stop $cid_retrievers && docker rm $cid_retrievers && sleep 1s
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
