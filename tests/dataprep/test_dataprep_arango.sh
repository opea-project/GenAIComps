#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

# Change this to point to the root of the project
WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

# Define all environment variables directly
export LOGFLAG="${LOGFLAG:-True}"
export no_proxy="${no_proxy:-noproxy,localhost,127.0.0.1,${ip_address}}"

# ArangoDB Configuration
export ARANGO_URL="${ARANGO_URL:-http://${ip_address}:8529}"
echo "ARANGO_URL: $ARANGO_URL"
export ARANGO_USERNAME="${ARANGO_USERNAME:-root}"
export ARANGO_PASSWORD="${ARANGO_PASSWORD:-test}"
export ARANGO_DB_NAME="${ARANGO_DB_NAME:-_system}"

# Dataprep Configuration
export DATAPREP_PORT="${DATAPREP_PORT:-6007}"
export DATAPREP_CHUNK_SIZE="${DATAPREP_CHUNK_SIZE:-500}"
export DATAPREP_CHUNK_OVERLAP="${DATAPREP_CHUNK_OVERLAP:-50}"
export DATAPREP_ARANGO_INSERT_ASYNC="${DATAPREP_ARANGO_INSERT_ASYNC:-false}"
export DATAPREP_ARANGO_USE_GRAPH_NAME="${DATAPREP_ARANGO_USE_GRAPH_NAME:-true}"
export DATAPREP_NODE_PROPERTIES="${DATAPREP_NODE_PROPERTIES:-}"
export DATAPREP_RELATIONSHIP_PROPERTIES="${DATAPREP_RELATIONSHIP_PROPERTIES:-}"
export DATAPREP_OPENAI_CHAT_ENABLED="${DATAPREP_OPENAI_CHAT_ENABLED:-false}"
export DATAPREP_OPENAI_EMBED_ENABLED="${DATAPREP_OPENAI_EMBED_ENABLED:-false}"
export DATAPREP_EMBED_NODES="${DATAPREP_EMBED_NODES:-true}"
export DATAPREP_EMBED_RELATIONSHIPS="${DATAPREP_EMBED_RELATIONSHIPS:-true}"
export DATAPREP_EMBED_SOURCE_DOCUMENTS="${DATAPREP_EMBED_SOURCE_DOCUMENTS:-true}"

# TEI Configuration
export TEI_PORT="${TEI_PORT:-6006}"
export EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-BAAI/bge-base-en-v1.5}"

# VLLM Configuration
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://test-comps-vllm-service:80}"
export VLLM_MODEL_ID="${VLLM_MODEL_ID:-Intel/neural-chat-7b-v3-3}"

function build_docker_images() {
	cd $WORKPATH
	echo "Working directory: $(pwd)"

	# Check if Dockerfile exists - updated path
	DOCKERFILE_PATH="comps/dataprep/src/Dockerfile"
	if [ ! -f "$DOCKERFILE_PATH" ]; then
		echo "Dockerfile not found at expected path: $DOCKERFILE_PATH"
		echo "Searching for Dockerfile..."
		find . -name "Dockerfile" | grep dataprep
		exit 1
	fi

	# Start ArangoDB with vector support
	docker run -d -p 8529:8529 \
		--name=test-comps-arango \
		-e ARANGO_ROOT_PASSWORD=$ARANGO_PASSWORD \
		arangodb/arangodb:3.12.4 \
		--experimental-vector-index=true

	# Build dataprep image
	echo "Building dataprep Docker image from $DOCKERFILE_PATH..."
	docker build --no-cache -t opea/dataprep:test \
		--build-arg https_proxy=$https_proxy \
		--build-arg http_proxy=$http_proxy \
		-f $DOCKERFILE_PATH .

	if [ $? -ne 0 ]; then
		echo "opea/dataprep built fail"
		exit 1
	else
		echo "opea/dataprep built successful"
	fi
}

function start_service() {
	# Create test network if it doesn't exist
	docker network create test-dataprep-network || true

	# Connect ArangoDB to the network
	docker network connect test-dataprep-network test-comps-arango || true

	# Create data directory if it doesn't exist
	mkdir -p $WORKPATH/data

	# Start VLLM service
	docker run -d \
		--name="test-comps-vllm-service" \
		--network test-dataprep-network \
		-p 9009:80 \
		-v $WORKPATH/data:/data \
		--shm-size=1g \
		-e no_proxy=$no_proxy \
		-e http_proxy=$http_proxy \
		-e https_proxy=$https_proxy \
		-e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN:-} \
		${REGISTRY:-opea}/vllm:${TAG:-latest} \
		--model ${VLLM_MODEL_ID} --host 0.0.0.0 --port 80

	echo "Started VLLM service with model: ${VLLM_MODEL_ID}"
	sleep 30s

	# Start TEI embedding service
	docker run -d \
		--name="test-comps-dataprep-tei-endpoint" \
		--network test-dataprep-network \
		-p $TEI_PORT:80 \
		-v $WORKPATH/data:/data \
		--pull always \
		ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 \
		--model-id ${EMBEDDING_MODEL_ID} --auto-truncate

	sleep 30s

	# Start dataprep service with all environment variables
	docker run -d \
		--name="test-comps-dataprep-server" \
		--network test-dataprep-network \
		-p $DATAPREP_PORT:5000 \
		--ipc=host \
		-e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_ARANGODB" \
		-e no_proxy=$no_proxy \
		-e http_proxy=$http_proxy \
		-e https_proxy=$https_proxy \
		-e ARANGO_URL=http://test-comps-arango:8529 \
		-e ARANGO_USERNAME=$ARANGO_USERNAME \
		-e ARANGO_PASSWORD=$ARANGO_PASSWORD \
		-e ARANGO_DB_NAME=$ARANGO_DB_NAME \
		-e ARANGO_INSERT_ASYNC=$DATAPREP_ARANGO_INSERT_ASYNC \
		-e ARANGO_USE_GRAPH_NAME=$DATAPREP_ARANGO_USE_GRAPH_NAME \
		-e TEI_EMBEDDING_ENDPOINT=http://test-comps-dataprep-tei-endpoint:80 \
		-e TEI_EMBED_MODEL=${EMBEDDING_MODEL_ID} \
		-e CHUNK_SIZE=$DATAPREP_CHUNK_SIZE \
		-e CHUNK_OVERLAP=$DATAPREP_CHUNK_OVERLAP \
		-e EMBED_SOURCE_DOCUMENTS=$DATAPREP_EMBED_SOURCE_DOCUMENTS \
		-e EMBED_NODES=$DATAPREP_EMBED_NODES \
		-e EMBED_RELATIONSHIPS=$DATAPREP_EMBED_RELATIONSHIPS \
		-e NODE_PROPERTIES=$DATAPREP_NODE_PROPERTIES \
		-e RELATIONSHIP_PROPERTIES=$DATAPREP_RELATIONSHIP_PROPERTIES \
		-e OPENAI_CHAT_ENABLED=$DATAPREP_OPENAI_CHAT_ENABLED \
		-e OPENAI_EMBED_ENABLED=$DATAPREP_OPENAI_EMBED_ENABLED \
		-e VLLM_API_KEY=$VLLM_API_KEY \
		-e VLLM_ENDPOINT=$VLLM_ENDPOINT \
		-e VLLM_MODEL_ID=$VLLM_MODEL_ID \
		-e LOGFLAG=$LOGFLAG \
		opea/dataprep:test

	sleep 1m
}

function validate_microservice() {
	# Create a test directory for files
	mkdir -p test_files

	# Create a test file with some structured content
	cat >test_files/test_doc.txt <<EOL
# Test Document
ArangoDB is the best database in the world. ArangoDB is a multi-model, open-source database with a flexible data model for documents, graphs, and key-values.
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
	# Stop and remove all containers
	for container in test-comps-dataprep-server test-comps-dataprep-tei-endpoint test-comps-vllm-service test-comps-arango; do
		if docker ps -q -f name=$container | grep -q .; then
			docker stop $container && docker rm $container
		fi
	done

	# Remove the test network if it exists
	docker network inspect test-dataprep-network >/dev/null 2>&1 && docker network rm test-dataprep-network
}

function main() {
	# Create log directory if it doesn't exist
	mkdir -p ${LOG_PATH}

	stop_docker
	build_docker_images
	start_service
	validate_microservice
	stop_docker
	echo y | docker system prune
}

main