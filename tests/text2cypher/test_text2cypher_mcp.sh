#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')

export NEO4J_USERNAME=${NEO4J_USERNAME:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4jtest"}
export NEO4J_URI=${NEO4J_URI:-"bolt://${ip_address}:7687"}
export NEO4J_URL=${NEO4J_URI}
export LOGFLAG=${LOGFLAG:-"False"}

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)

    docker build --no-cache -t opea/text2cypher-gaudi:mcp-test --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/text2cypher/src/Dockerfile.intel_hpu .
    if [ $? -ne 0 ]; then
        echo "opea/text2cypher-gaudi built fail"
        exit 1
    else
        echo "opea/text2cypher-gaudi built successful"
    fi
}

function start_service() {
    echo "Starting text2cypher service with MCP enabled..."
    cd $WORKPATH
    export TEXT2CYPHER_PORT=9097
    export TAG=mcp-test
    export ENABLE_MCP=true
    cd comps/text2cypher/deployment/docker_compose/

    # Start services including Neo4j
    docker compose up -d

    # Wait for services to be ready
    echo "Waiting for services to start..."
    sleep 30s

    # Check if Neo4j is healthy
    docker ps | grep neo4j-apoc
    if [ $? -ne 0 ]; then
        echo "Neo4j service failed to start"
        docker logs neo4j-apoc
        exit 1
    fi

    # Check if text2cypher service is running
    docker ps | grep text2cypher-gaudi-container
    if [ $? -ne 0 ]; then
        echo "text2cypher service failed to start"
        docker logs text2cypher-gaudi-container
        exit 1
    fi
}

function validate_microservice() {
    pip install mcp
    python3 ${WORKPATH}/tests/text2cypher/validate_svc_with_mcp.py $ip_address $TEXT2CYPHER_PORT
    if [ $? -ne 0 ]; then
        echo "MCP validation failed"
        docker logs text2cypher-gaudi-container
        docker logs neo4j-apoc
        exit 1
    fi
}

function stop_docker() {
    docker ps -a --filter "name=text2cypher-gaudi-container" --filter "name=neo4j-apoc" --format "{{.Names}}" | xargs -r docker stop
    docker ps -a --filter "name=text2cypher-gaudi-container" --filter "name=neo4j-apoc" --format "{{.Names}}" | xargs -r docker rm
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
