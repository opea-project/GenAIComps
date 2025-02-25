#!/usr/bin/env bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT=11103
LLM_ENDPOINT_PORT=10510
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg no_proxy=$no_proxy --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
    docker pull ghcr.io/huggingface/tgi-gaudi:2.3.1
    docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
}

function start_service() {
    service_name="neo4j-apoc tei-embedding-serving tgi-gaudi-server dataprep-neo4j-llamaindex"
    export host_ip=${ip_address}
    export NEO4J_PORT1=7474   # 11631
    export NEO4J_PORT2=7687   # 11632
    export NEO4J_AUTH="neo4j/neo4jtest"
    export NEO4J_URL="bolt://${ip_address}:${NEO4J_PORT2}"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="neo4jtest"
    export NEO4J_apoc_export_file_enabled=true
    export NEO4J_apoc_import_file_use__neo4j__config=true
    export NEO4J_PLUGINS=\[\"apoc\"\]
    export TEI_EMBEDDER_PORT=12006
    export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export EMBED_MODEL=${EMBEDDING_MODEL_ID}
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${TEI_EMBEDDER_PORT}"
    export LLM_ENDPOINT_PORT=10510
    export TGI_LLM_ENDPOINT="http://${ip_address}:${LLM_ENDPOINT_PORT}"
    export MAX_INPUT_TOKENS=4096
    export MAX_TOTAL_TOKENS=8192

    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate_microservice() {
    # validate neo4j-apoc
    call_curl "http://${ip_address}:${NEO4J_PORT1}" 'Content-Type: application/json'
    check_result "neo4j-apoc" "" neo4j-apoc ${LOG_PATH}/dataprep_neo4j-apoc.log

    sleep 1m  # retrieval can't curl as expected, try to wait for more time
    # tgi for llm service
    call_curl "http://${ip_address}:${LLM_ENDPOINT_PORT}/generate" 'Content-Type: application/json' \
	      -XPOST \
	      -d '{"inputs":"WhatisDeepLearning?","parameters":{"max_new_tokens":17,"do_sample":true}}'
    check_result "tgi-gaudi-service" "generated_text" tgi-gaudi-server ${LOG_PATH}/dataprep_neo4j-tgi-gaudi.log

    # test /v1/dataprep/ingest upload file
    ingest_doc ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

    ingest_docx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

    ingest_pdf ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

    ingest_pptx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

    ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

    # test /v1/dataprep/ingest upload link
    ingest_external_link ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-neo4j-llamaindex ${LOG_PATH}/dataprep_neo4j.log

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-neo4j*")
    if [[ ! -z "$cid" ]]; then
        docker stop $cid && docker rm $cid && sleep 1s
    fi
    cid_db=$(docker ps -aq --filter "name=neo4j-apoc" --filter "name=tgi-gaudi-server")
    if [[ ! -z "$cid_db" ]]; then
        docker stop $cid_db && docker rm $cid_db && sleep 1s
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
