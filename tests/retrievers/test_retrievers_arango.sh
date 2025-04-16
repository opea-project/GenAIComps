#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
export host_ip=$(hostname -I | awk '{print $1}')
service_name="retriever-arangodb"

export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:6060"
export EMBEDDING_MODEL_ID=BAAI/bge-base-en-v1.5
export service_name="retriever-arangodb"
export ARANGO_URL=${ARANGO_URL:-"http://${host_ip}:8529"}
export ARANGO_USERNAME=${ARANGO_USERNAME:-"root"}
export ARANGO_PASSWORD=${ARANGO_PASSWORD:-"test"}
export ARANGO_DB_NAME=${ARANGO_DB_NAME:-"_system"}
export ARANGO_COLLECTION_NAME=${ARANGO_COLLECTION_NAME:-"test"}

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/retriever:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever built fail"
        exit 1
    else
        echo "opea/retriever built successful"
    fi
}

function start_service() {

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 1m

}

function validate_microservice() {
    export PATH="${HOME}/miniforge3/bin:$PATH"
    source activate
    URL="http://${host_ip}:7000/v1/retrieval"

     # Create ARANGO_COLLECTION_NAME
    curl -X POST --header 'accept: application/json' \
    --header 'Content-Type: application/json' \
    --data '{"name": "'${ARANGO_COLLECTION_NAME}'", "type": 2, "waitForSync": true}' \
    "${ARANGO_URL}/_db/${ARANGO_DB_NAME}/_api/collection" \
    -u ${ARANGO_USERNAME}:${ARANGO_PASSWORD}

    # Insert data into arango: {text: "test", embedding: [0.1, 0.2, 0.3, 0.4, 0.5]}
    curl -X POST --header 'accept: application/json' \
    --header 'Content-Type: application/json' \
    --data '{"text": "test", "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}' \
    "${ARANGO_URL}/_db/${ARANGO_DB_NAME}/_api/document/${ARANGO_COLLECTION_NAME}" \
    -u ${ARANGO_USERNAME}:${ARANGO_PASSWORD}

    sleep 1m

    test_embedding="[0.1, 0.2, 0.3, 0.4, 0.5]"
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"test\",\"embedding\":${test_embedding}}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content is as expected."
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs ${service_name} >> ${LOG_PATH}/retriever.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs ${service_name} >> ${LOG_PATH}/retriever.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/third_parties/arangodb/deployment/docker_compose/
    docker compose -f compose.yaml down -v --remove-orphans

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down  ${service_name} --remove-orphans

    cid=$(docker ps -aq --filter "name=tei-embedding-serving")
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
