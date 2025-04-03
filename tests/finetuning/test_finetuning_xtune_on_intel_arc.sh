#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
finetuning_service_port=8015
ray_port=8265
export DATA=${DATA:-/data}
service_name="finetuning-xtune"

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build -t opea/finetuning-xtune:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg HF_TOKEN=$HF_TOKEN --build-arg DATA=$DATA -f comps/finetuning/src/Dockerfile.xtune .
    if [ $? -ne 0 ]; then
        echo "opea/finetuning-xtune built fail"
        exit 1
    else
        echo "opea/finetuning-xtune built successful"
    fi
}

function start_service() {
    export no_proxy="localhost,127.0.0.1,"${ip_address}
    cd $WORKPATH/comps/finetuning/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > start_services_with_compose.log
    sleep 1m
}

function validate_upload() {
    local URL="$1"
    local SERVICE_NAME="$2"
    local DOCKER_NAME="$3"
    local EXPECTED_PURPOSE="$4"
    local EXPECTED_FILENAME="$5"

    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -F "file=@./$EXPECTED_FILENAME" -F purpose="$EXPECTED_PURPOSE" -H 'Content-Type: multipart/form-data' "$URL")
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')

    # Parse the JSON response
    purpose=$(echo "$RESPONSE_BODY" | jq -r '.purpose')
    filename=$(echo "$RESPONSE_BODY" | jq -r '.filename')

    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_upload_file.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."
    fi

    # Check if the parsed values match the expected values
    if [[ "$purpose" != "$EXPECTED_PURPOSE" || "$filename" != "$EXPECTED_FILENAME" ]]; then
        echo "[ $SERVICE_NAME ] Content does not match the expected result: $RESPONSE_BODY"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_upload_file.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] Content is as expected."
    fi

    sleep 10s
}

function validate_finetune() {
    local URL="$1"
    local SERVICE_NAME="$2"
    local DOCKER_NAME="$3"
    local EXPECTED_DATA="$4"
    local INPUT_DATA="$5"

    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -H 'Content-Type: application/json' -d "$INPUT_DATA" "$URL")
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')
    FINTUNING_ID=$(echo "$RESPONSE_BODY" | jq -r '.id')

    # Parse the JSON response
    purpose=$(echo "$RESPONSE_BODY" | jq -r '.purpose')
    filename=$(echo "$RESPONSE_BODY" | jq -r '.filename')
    echo "purpose : $purpose"
    echo "RESPONSE_BODY : $RESPONSE_BODY"
    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_create.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."
    fi

    # Check if the parsed values match the expected values
    if [[ "$RESPONSE_BODY" != *"$EXPECTED_DATA"* ]]; then
        echo "[ $SERVICE_NAME ] Content does not match the expected result: $RESPONSE_BODY"
        docker logs $DOCKER_NAME >> ${LOG_PATH}/finetuning-server_create.log
        exit 1
    else
        echo "[ $SERVICE_NAME ] Content is as expected."
    fi

    sleep 10s

    # check finetuning job status
    URL="$URL/retrieve"
    for((i=1;i<=10;i++));
    do
	HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -H "Content-Type: application/json" -d '{"fine_tuning_job_id": "'$FINTUNING_ID'"}' "$URL")
	echo $HTTP_RESPONSE
	RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')
	STATUS=$(echo "$RESPONSE_BODY" | jq -r '.status')
	if [[ "$STATUS" == "succeeded" ]]; then
	    echo "training: succeeded."
	    break
	elif [[ "$STATUS" == "failed" ]]; then
	    echo "training: failed."
	    exit 1
	else
	    echo "training: '$STATUS'"
        sleep 1m
	fi
    done
}

function validate_microservice() {
    cd $LOG_PATH
    export no_proxy="localhost,127.0.0.1,"${ip_address}

    ##########################
    #    general test         #
    ##########################
    # test /v1/dataprep upload file
    cat $WORKPATH/tests/finetuning/json_data.json > $LOG_PATH/test_data.json
    validate_upload \
        "http://${ip_address}:$finetuning_service_port/v1/files" \
        "general - upload" \
        "finetuning-xtune" \
        "fine-tune" \
        "test_data.json"

    # test /v1/fine_tuning/jobs
    validate_finetune \
        "http://${ip_address}:$finetuning_service_port/v1/fine_tuning/jobs" \
        "general - finetuning" \
        "finetuning-xtune" \
        '{"id":"ft-job' \
        '{"training_file": "","model": "vit_b16", "General":{"xtune_config":{"tool":"clip","device":"XPU", "dataset_root":"/home/data", "trainer": "clip_adapter_hf", "dataset":"caltech101", "model":"vit_b16"}}}'




}

function stop_docker() {
    cd $WORKPATH/comps/finetuning/deployment/docker_compose
    docker compose -f compose.yaml down ${service_name} --remove-orphans
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
