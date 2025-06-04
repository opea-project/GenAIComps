#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="textgen-service-vllm-gaudi"

function build_docker_images() {
    cd $WORKPATH
    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork/
    VLLM_VER=v0.6.6.post1+Gaudi-1.20.0
    echo "Check out vLLM tag ${VLLM_VER}"
    git checkout ${VLLM_VER} &> /dev/null
    docker build --no-cache -f Dockerfile.hpu -t ${REGISTRY:-opea}/vllm-gaudi:${TAG:-latest} --shm-size=128g .
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi built fail"
        exit 1
    else
        echo "opea/vllm-gaudi built successful"
    fi

    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/llm-textgen:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/llm-textgen built fail"
        exit 1
    else
        echo "opea/llm-textgen built successful"
    fi
}

function start_service() {
    export LLM_ENDPOINT_PORT=12110  # 12100-12199
    export TEXTGEN_PORT=10510 #10500-10599
    export host_ip=${host_ip}
    # Check and set important environment variables with defaults if not set
    if [ -z "${HF_TOKEN}" ]; then
        echo "WARNING: HF_TOKEN is not set. This may cause model download failures."
    fi
    export HF_TOKEN=${HF_TOKEN:-""}
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
    export VLLM_SKIP_WARMUP=true
    export LOGFLAG=True
    export DATA_PATH=${model_cache:-./data}
    
    # Create log directory
    mkdir -p ${LOG_PATH}/llms
    
    # Print important environment settings
    echo "Environment variables:"
    echo "  host_ip=${host_ip}"
    echo "  LLM_ENDPOINT=${LLM_ENDPOINT}"
    echo "  LLM_MODEL_ID=${LLM_MODEL_ID}" 
    echo "  DATA_PATH=${DATA_PATH}"
    echo "  HF_TOKEN is $(if [ -z "${HF_TOKEN}" ]; then echo "NOT "; fi)set"

    cd $WORKPATH/comps/llms/deployment/docker_compose
    
    # Display the compose file content for debugging
    echo "Docker compose configuration:"
    echo "=========================================================="
    cat compose_text-generation.yaml | tee ${LOG_PATH}/llms/compose_text-generation.yaml.log
    echo "=========================================================="
    
    # Start services in the background but capture logs
    docker compose -f compose_text-generation.yaml up ${service_name} -d | tee ${LOG_PATH}/llms/start_services_with_compose.log
    
    echo "Waiting for services to start..."
    # Immediately start capturing logs from the container to avoid missing early errors
    docker compose -f compose_text-generation.yaml logs -f vllm-gaudi-server > ${LOG_PATH}/llms/vllm-gaudi-server-startup.log 2>&1 &
    LOG_PID=$!
    
    # Wait a bit for services to start
    sleep 30s
    
    # Stop background log capture
    kill $LOG_PID || true
    
    # Check if vllm-gaudi-server container is running
    if ! docker ps | grep -q "vllm-gaudi-server"; then
        echo "vllm-gaudi-server failed to start. Showing logs:"
        
        # Get logs even from containers that have exited
        docker compose -f compose_text-generation.yaml logs vllm-gaudi-server | tee ${LOG_PATH}/llms/vllm-gaudi-server-error.log
        
        # Try to get more details about the container's exit
        echo "Container exit details:"
        docker inspect vllm-gaudi-server --format='{{json .State}}' | jq . | tee ${LOG_PATH}/llms/vllm-gaudi-server-state.log
        
        exit 1
    fi
}

function validate_services() {
    local URL="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_NAME="$3"
    local DOCKER_NAME="$4"
    local INPUT_DATA="$5"

    echo "==========================================="
    echo "[ $SERVICE_NAME ] Testing endpoint: $URL"
    echo "[ $SERVICE_NAME ] Input data: $INPUT_DATA"
    
    # Create log file for this test
    local LOG_FILE="${LOG_PATH}/llms/${SERVICE_NAME}-validation.log"
    echo "Validation started at $(date)" > $LOG_FILE
    
    # First check if the container is still running
    if ! docker ps | grep -q "$DOCKER_NAME"; then
        echo "[ $SERVICE_NAME ] ERROR: Container $DOCKER_NAME is not running" | tee -a $LOG_FILE
        echo "Container logs:" | tee -a $LOG_FILE
        docker logs $DOCKER_NAME &>> $LOG_FILE
        echo "Container status:" | tee -a $LOG_FILE
        docker inspect $DOCKER_NAME --format='{{json .State}}' | jq . &>> $LOG_FILE
        return 1
    fi

    # Send request with timeout and full response capture
    echo "Sending request to $URL..." | tee -a $LOG_FILE
    local RESPONSE_FILE="${LOG_PATH}/llms/${SERVICE_NAME}-response.txt"
    local HTTP_STATUS=$(curl -v -s -o $RESPONSE_FILE -w "%{http_code}" -X POST -d "$INPUT_DATA" \
                      -H 'Content-Type: application/json' --max-time 30 "$URL" 2>> $LOG_FILE)
    echo "Response status code: $HTTP_STATUS" | tee -a $LOG_FILE
    
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..." | tee -a $LOG_FILE

        cat $RESPONSE_FILE >> $LOG_FILE
        local CONTENT=$(cat $RESPONSE_FILE)

        if echo "$CONTENT" | grep -q "$EXPECTED_RESULT"; then
            echo "[ $SERVICE_NAME ] Content is as expected." | tee -a $LOG_FILE
            return 0
        else
            echo "[ $SERVICE_NAME ] Content does not match the expected result:" | tee -a $LOG_FILE
            echo "$CONTENT" | tee -a $LOG_FILE
            echo "Expected to find: $EXPECTED_RESULT" | tee -a $LOG_FILE
            echo "Container logs:" | tee -a $LOG_FILE
            docker logs $DOCKER_NAME &>> $LOG_FILE
            return 1
        fi
    else
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS" | tee -a $LOG_FILE
        
        if [ -f "$RESPONSE_FILE" ]; then
            echo "Response body:" | tee -a $LOG_FILE
            cat $RESPONSE_FILE | tee -a $LOG_FILE
        fi
        
        echo "Container logs:" | tee -a $LOG_FILE
        docker logs $DOCKER_NAME &>> $LOG_FILE
        return 1
    fi
    sleep 1s
}

function validate_microservices() {
    URL="http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions"

    # Check service health first
    echo "Checking if services are running and healthy..."
    docker ps | grep -E 'vllm-gaudi-server|textgen-service' | tee ${LOG_PATH}/llms/containers-running.log
    
    # vllm
    echo "Validate vllm..."
    validate_services \
        "${LLM_ENDPOINT}/v1/completions" \
        "text" \
        "vllm-gaudi-server" \
        "vllm-gaudi-server" \
        '{"model": "Intel/neural-chat-7b-v3-3", "prompt": "What is Deep Learning?", "max_tokens": 32, "temperature": 0}'

    # textgen
    echo "Validate textgen with string messages input..."
    validate_services \
        "$URL" \
        "text" \
        "textgen-service-vllm-gaudi" \
        "textgen-service-vllm-gaudi" \
        '{"model": "Intel/neural-chat-7b-v3-3", "messages": "What is Deep Learning?", "max_tokens":17, "stream":false}'

    echo "Validate textgen with dict messages input..."
    validate_services \
        "$URL" \
        "content" \
        "textgen-service-vllm-gaudi" \
        "textgen-service-vllm-gaudi" \
        '{"model": "Intel/neural-chat-7b-v3-3", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17, "stream":false}'
}

function validate_microservice_with_openai() {
    python3 ${WORKPATH}/tests/utils/validate_svc_with_openai.py "$host_ip" "$TEXTGEN_PORT" "llm"
    if [ $? -ne 0 ]; then
        docker logs vllm-gaudi-server >> ${LOG_PATH}/llm--gaudi.log
        docker logs textgen-service-vllm-gaudi >> ${LOG_PATH}/llm-server.log
        exit 1
    fi
}

function stop_docker() {
    cd $WORKPATH/comps/llms/deployment/docker_compose
    
    # Capture logs before stopping in case there were issues
    echo "Capturing final logs before shutdown..."
    docker compose -f compose_text-generation.yaml logs vllm-gaudi-server > ${LOG_PATH}/llms/vllm-gaudi-server-final.log 2>&1
    docker compose -f compose_text-generation.yaml logs textgen-service-vllm-gaudi > ${LOG_PATH}/llms/textgen-service-final.log 2>&1
    
    # Show container status
    docker ps -a | grep -E 'vllm|textgen' > ${LOG_PATH}/llms/container-status-final.log 2>&1
    
    # Stop containers
    echo "Stopping all containers..."
    docker compose -f compose_text-generation.yaml down --remove-orphans
}

function main() {
    # Create log directory
    mkdir -p ${LOG_PATH}/llms
    
    # Log system information
    echo "System information:" | tee ${LOG_PATH}/llms/system_info.log
    uname -a >> ${LOG_PATH}/llms/system_info.log
    free -h >> ${LOG_PATH}/llms/system_info.log
    lscpu >> ${LOG_PATH}/llms/system_info.log
    
    # Check if HPUs are available (for Gaudi servers)
    if [ -x "$(command -v hlkd_admin)" ]; then
        echo "HPU information:" | tee -a ${LOG_PATH}/llms/system_info.log
        hlkd_admin status >> ${LOG_PATH}/llms/system_info.log 2>&1 || echo "Could not get HPU status" >> ${LOG_PATH}/llms/system_info.log
    fi
    
    # Check Docker status
    docker info | tee ${LOG_PATH}/llms/docker_info.log
    
    # Verify environment
    echo "Environment variables:" | tee ${LOG_PATH}/llms/environment.log
    env | grep -E 'HF_|TOKEN|LLM|MODEL|PATH|VLLM|TEXTGEN|proxy|REGISTRY|TAG' | sort >> ${LOG_PATH}/llms/environment.log
    
    # Clean up any previous docker containers
    stop_docker
    
    # Set up error handling
    set +e  # Continue on error, but track it
    ERROR=0

    # Build and start containers
    build_docker_images
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build Docker images" | tee -a ${LOG_PATH}/llms/error.log
        ERROR=1
    else
        pip install --no-cache-dir openai pydantic
        start_service
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to start services" | tee -a ${LOG_PATH}/llms/error.log
            ERROR=1
        else
            # Only validate if services started successfully
            validate_microservices
            if [ $? -ne 0 ]; then
                echo "ERROR: Microservice validation failed" | tee -a ${LOG_PATH}/llms/error.log
                ERROR=1
            else
                validate_microservice_with_openai
                if [ $? -ne 0 ]; then
                    echo "ERROR: OpenAI validation failed" | tee -a ${LOG_PATH}/llms/error.log
                    ERROR=1
                fi
            fi
        fi
    fi
    
    # Always try to stop and clean up
    stop_docker
    echo y | docker system prune
    
    # Exit with the appropriate status
    echo "Test completed with status: $ERROR" | tee -a ${LOG_PATH}/llms/status.log
    return $ERROR
}

main
