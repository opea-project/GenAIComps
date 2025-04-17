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
    export VLLM_MODEL_ID="${VLLM_MODEL_ID:-Intel/neural-chat-7b-v3-3}"
    export LLM_MODEL_ID="${LLM_MODEL_ID:-Intel/neural-chat-7b-v3-3}"
    export HF_TOKEN="${HF_TOKEN:-EMPTY}"
    export HuggingFaceHub_API_TOKEN="${HF_TOKEN:-EMPTY}"

    export LOGFLAG=true
    
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    # Ensure host_ip and LLM_ENDPOINT_PORT are available to docker compose
    docker compose up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    # Debug time
    sleep 1m

    check_healthy "dataprep-arangodb" || exit 1
}

function validate_microservice() {
	# Create a test directory for files
	mkdir -p test_files
	# Create a test file with some structured content
	cat >test_files/test_doc.txt <<EOL
# Test Document
ArangoDB is a multi-model, open-source database with a flexible data model for documents, graphs, and key-values.
EOL
	# Test file upload
	echo "Testing ingest endpoint..."
	HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
		-X POST \
		-F "files=@test_files/test_doc.txt" \
		-F "chunk_size=$DATAPREP_CHUNK_SIZE" \
		-F "chunk_overlap=$DATAPREP_CHUNK_OVERLAP" \
		-F "process_table=false" \
		-F "table_strategy=fast" \
		"http://localhost:$DATAPREP_PORT/v1/dataprep/ingest")
	if [ "$HTTP_STATUS" -eq 200 ]; then
		echo "[ dataprep ] Ingest endpoint test passed"
		# Capture the full response for logging
		INGEST_RESPONSE=$(curl -s \
			-X POST \
			-F "files=@test_files/test_doc.txt" \
			-F "chunk_size=$DATAPREP_CHUNK_SIZE" \
			-F "chunk_overlap=$DATAPREP_CHUNK_OVERLAP" \
			-F "process_table=false" \
			-F "table_strategy=fast" \
			"http://localhost:$DATAPREP_PORT/v1/dataprep/ingest" | tee ${LOG_PATH}/dataprep_ingest.log)
		echo "Ingest response: $INGEST_RESPONSE"
		# Test get endpoint
		echo "Testing get endpoint..."
		GET_RESPONSE=$(curl -s -X POST "http://localhost:$DATAPREP_PORT/v1/dataprep/get" | tee ${LOG_PATH}/dataprep_get.log)
		GET_STATUS=$?
		if [ "$GET_STATUS" -eq 0 ]; then
			echo "[ dataprep ] Get endpoint test passed"
			echo "Get response: $GET_RESPONSE"
			# Check if the response has valid content (should be an array)
			if echo "$GET_RESPONSE" | grep -q '\[.*\]' || echo "$GET_RESPONSE" | grep -q 'name'; then
				echo "[ dataprep ] Get response is valid"
			else
				echo "[ dataprep ] Get response is not valid: $GET_RESPONSE"
				docker logs test-comps-dataprep-server >>${LOG_PATH}/dataprep.log
				docker logs test-comps-dataprep-tei-endpoint >>${LOG_PATH}/tei.log
				exit 1
			fi
		else
			echo "[ dataprep ] Get endpoint test failed"
			docker logs test-comps-dataprep-server >>${LOG_PATH}/dataprep.log
			docker logs test-comps-dataprep-tei-endpoint >>${LOG_PATH}/tei.log
			exit 1
		fi
		# Verify data in ArangoDB
		echo "Verifying ArangoDB data..."
		GRAPH_CHECK=$(curl -s \
			"http://localhost:8529/_db/${ARANGO_DB_NAME}/_api/gharial" \
			-u ${ARANGO_USERNAME}:${ARANGO_PASSWORD} | tee ${LOG_PATH}/arango_graph.log)
		if echo "$GRAPH_CHECK" | grep -q "GRAPH"; then
			echo "[ dataprep ] Graph verification passed"
		else
			echo "[ dataprep ] Graph verification failed"
			docker logs test-comps-dataprep-server >>${LOG_PATH}/dataprep.log
			exit 1
		fi
	else
		echo "[ dataprep ] Ingest endpoint test failed with status $HTTP_STATUS"
		docker logs test-comps-dataprep-server >>${LOG_PATH}/dataprep.log
		docker logs test-comps-dataprep-tei-endpoint >>${LOG_PATH}/tei.log
		exit 1
	fi
	# Clean up test files
	rm -rf test_files
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
