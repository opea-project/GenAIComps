#!/usr/bin/env bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
export TEXT2CYPHER_PORT=11801
export TAG='comps'
export HF_TOKEN=${HF_TOKEN}
export host_ip=${ip_address}
export NEO4J_PORT1=7474
export NEO4J_PORT2=7687
export NEO4J_URI="bolt://${host_ip}:${NEO4J_PORT2}"
export NEO4J_URL="bolt://${host_ip}:${NEO4J_PORT2}"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="neo4jtest"
export no_proxy="localhost,127.0.0.1,"${host_ip}
export LOGFLAG=True

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/text2query-cypher:$TAG --build-arg no_proxy=$no_proxy --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2query/src/Dockerfile.cypher.intel_hpu .
    if [ $? -ne 0 ]; then
        echo "opea/text2query-cypher built fail"
        exit 1
    else
        echo "opea/text2query-cypher built successful"
    fi
}

function start_service() {
    docker compose down
    unset http_proxy
    service_name="text2query-cypher-gaudi"
    export NEO4J_AUTH="neo4j/neo4jtest"
    export NEO4J_apoc_export_file_enabled=true
    export NEO4J_apoc_import_file_use__neo4j__config=true
    export NEO4J_PLUGINS=\[\"apoc\"\]

    cd $WORKPATH/comps/text2query/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate_neo4j_service() {
    local URL="${ip_address}:7474"
    local EXPECTED_RESULT="200 OK"
    local SERVICE_NAME="neo4j-apoc"
    local CONTAINER_NAME="neo4j-apoc"

    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" "$URL")
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')

    docker logs ${CONTAINER_NAME}

    # check response status
    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200."
    fi

    sleep 1m
}

function validate_text2cypher_service() {
    local SERVICE_NAME="text2query-cypher-gaudi"
    local CONTAINER_NAME="text2query-cypher-gaudi-server"

    result=$(http_proxy="" curl http://${ip_address}:${TEXT2CYPHER_PORT}/v1/text2query\
        -X POST \
        -d '{"query": "what are the symptoms for Diabetes?","conn_type": "cypher","conn_user": "'${NEO4J_USERNAME}'","conn_password": "neo4jtest","conn_url": "'${NEO4J_URL}'","conn_dialect": "neo4j" }' \
        -H 'Content-Type: application/json')

    if [[ ${#result} -gt 0 ]]; then
        echo $result
        echo "Result correct."
    else
        echo "Result wrong. Received was $result"
        docker logs ${CONTAINER_NAME}
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=text2query-cypher-gaudi*")
    if [[ ! -z "$cid" ]]; then
        docker stop $cid && docker rm $cid && sleep 1s
    fi
    cid_db=$(docker ps -aq --filter "name=neo4j-apoc")
    if [[ ! -z "$cid_db" ]]; then
        docker stop $cid_db && docker rm $cid_db && sleep 1s
    fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    #validate_neo4j_service
    validate_text2cypher_service

    stop_docker
    echo y | docker system prune

}

main
