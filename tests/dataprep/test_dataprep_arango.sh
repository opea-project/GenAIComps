#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

export WORKPATH=$(dirname "$PWD")
export LOG_PATH="$WORKPATH/tests"
export ip_address=$(hostname -I | awk '{print $1}')
export DATAPREP_PORT=${DATAPREP_PORT:-6007}
service_name="dataprep-arangodb"
export TAG="latest"
export DATA_PATH=${model_cache}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build --no-cache -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function start_service() {

    export ARANGO_URL="${ARANGO_URL:-http://arango-vector-db:8529}"
    export ARANGO_USERNAME="${ARANGO_USERNAME:-root}"
    export ARANGO_PASSWORD="${ARANGO_PASSWORD:-test}"
    export ARANGO_DB_NAME="${ARANGO_DB_NAME:-_system}"

    # Define host_ip *before* first use (if needed elsewhere)
    export host_ip=$(hostname -I | awk '{print $1}')

    # TEI Configuration
    export TEI_PORT="${TEI_PORT:-6006}" # This port seems unused if endpoint is defined
    export TEI_EMBEDDER_PORT=${TEI_EMBEDDER_PORT:-8080} # Define default TEI port if not set
    export EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-BAAI/bge-base-en-v1.5}"
    # Use the correct *internal* port (80) for TEI service communication
    export TEI_EMBEDDING_ENDPOINT="${TEI_EMBEDDING_ENDPOINT:-http://tei-embedding-serving:80}"

    # VLLM Configuration
    # host_ip is already defined above
    export LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-8008}
    export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
    export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://vllm-server:80}"
    export VLLM_MODEL_ID="${VLLM_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
    export LLM_MODEL_ID="${LLM_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
    export HF_TOKEN="${HF_TOKEN:-EMPTY}"

    export LOGFLAG=true


    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    # Ensure host_ip and LLM_ENDPOINT_PORT are available to docker compose
    docker compose up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    # Debug time
    sleep 2m

    check_healthy "dataprep-arangodb" || exit 1
}

function validate_microservice() {
	# NOTE: Due to due to the requirements of using an
	# LLM to extract Entities & Relationships as part of the
	# ArangoDB Dataprep Service, and the lack of performance
	# in the LLM selected to host via vllm on CI environments,
	# some of these upload tests will take too long to fully complete.
	# For example, ingest_dataprep.pdf is 107 pages.
	# Using the base Intel/neural-chat-7b-v3-3 model for a PDF
	# size like this will take around 5-10 minutes to extract Entities
	# & Relationships, per chunk.
	# Reference:
	# https://github.com/opea-project/GenAIComps/pull/1558#discussion_r2048402988

    # test /v1/dataprep/ingest upload file
    # ingest_doc ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - doc" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # ingest_docx ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - docx" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # ingest_pdf ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - pdf" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # ingest_ppt ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - ppt" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_upload_file.log

    # ingest_pptx ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - pptx" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    ingest_txt ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - upload - txt" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # ingest_xlsx ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - xlsx" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # # test /v1/dataprep/ingest upload link
    # ingest_external_link ${ip_address} ${DATAPREP_PORT}
    # check_result "dataprep - upload - link" "Data preparation succeeded" dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # test /v1/dataprep/get
    get_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - get" '{"name":' dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log

    # test /v1/dataprep/delete
    delete_all ${ip_address} ${DATAPREP_PORT}
    check_result "dataprep - del" '{"status":true}' dataprep-arangodb ${LOG_PATH}/dataprep_arangodb.log
}

function stop_docker() {
    cd $WORKPATH/comps/third_parties/arangodb/deployment/docker_compose/
    docker compose -f compose.yaml down --remove-orphans

    cd $WORKPATH/comps/dataprep/deployment/docker_compose
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
