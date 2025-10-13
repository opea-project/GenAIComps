#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../llms" &> /dev/null && pwd )"
source "${SCRIPT_DIR}/llm_utils.sh"

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"
export DATA_PATH=${model_cache}

WORKPATH=$(dirname "$PWD")
host_ip=$(hostname -I | awk '{print $1}')
LOG_PATH="$WORKPATH/tests"
service_name="arbPostHearingAssistant-tgi"

function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t ${REGISTRY:-opea}/arb-post-hearing-assistant:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/arb_post_hearing_assistant/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/arb-post-hearing-assistant built fail"
        exit 1
    else
        echo "opea/arb-post-hearing-assistant built successful"
    fi
}

function start_service() {
    local offline=${1:-false}
    export host_ip=${host_ip}
    export LLM_ENDPOINT_PORT=12105  # 12100-12199
    export OPEA_ARB_POSTHEARING_ASSISTANT_PORT=10505 #10500-10599
    export HF_TOKEN=${HF_TOKEN}
    export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export LLM_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
    export MAX_INPUT_TOKENS=2048
    export MAX_TOTAL_TOKENS=4096
    export LOGFLAG=True

    service_name="arbPostHearingAssistant-tgi"
    if [[ "$offline" == "true" ]]; then
        service_name="arbPostHearingAssistant-tgi-offline"
        export offline_no_proxy="${host_ip}"
        prepare_models ${DATA_PATH} ${LLM_MODEL_ID} gpt2
    fi
    cd $WORKPATH/comps/arb_post_hearing_assistant/deployment/docker_compose
    docker compose -f arb_post_hearing_assistant-compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log

    sleep 30s
}

function validate_services() {
    local URL="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_NAME="$3"
    local DOCKER_NAME="$4"
    local INPUT_DATA="$5"

    local HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL")

    echo "==========================================="

    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."

        local CONTENT=$(curl -s -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/${SERVICE_NAME}.log)

        echo $CONTENT

        if echo "$CONTENT" | grep -q "$EXPECTED_RESULT"; then
            echo "[ $SERVICE_NAME ] Content is as expected."
        else
            echo "[ $SERVICE_NAME ] Content does not match the expected result"
            docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_NAME}.log
            exit 1
        fi
    else
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_NAME}.log
        docker exec ${DOCKER_NAME} env
        exit 1
    fi
    sleep 1s
}

function validate_microservices() {
    URL="http://${host_ip}:${OPEA_ARB_POSTHEARING_ASSISTANT_PORT}/v1/arb-post-hearing"

    echo "Validate tgi..."
    validate_services \
        "${LLM_ENDPOINT}/generate" \
        "generated_text" \
        "tgi-server" \
        "tgi-server" \
        '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}'

    echo "Validate Transcript..."
    validate_services \
        "$URL" \
        'text' \
        "arbPostHearingAssistant" \
        "arbPostHearingAssistant" \
        '{"messages": "[10:00 AM] Arbitrator Hon. Rebecca Lawson: Good morning. This hearing is now in session for Case No. ARB/2025/0917. Lets begin with appearances. [10:01 AM] Attorney Michael Grant for Mr. Jonathan Reed: Good morning Your Honor. I represent the claimant Mr. Jonathan Reed. [10:01 AM] Attorney Lisa Chen for Ms. Rachel Morgan: Good morning. I represent the respondent Ms. Rachel Morgan. [10:03 AM] Arbitrator Hon. Rebecca Lawson: Thank you. Lets proceed with Mr. Reeds opening statement. [10:04 AM] Attorney Michael Grant: Ms. Morgan failed to deliver services as per the agreement dated March 15 2023. We have submitted relevant documentation including email correspondence and payment records. The delay caused substantial financial harm to our client. [10:15 AM] Attorney Lisa Chen: We deny any breach of contract. The delays were due to regulatory issues outside our control. Furthermore Mr. Reed did not provide timely approvals which contributed to the delay. [10:30 AM] Arbitrator Hon. Rebecca Lawson: Lets turn to Clause Z of the agreement. Id like both parties to submit written briefs addressing the applicability of the force majeure clause and the timeline of approvals. [11:00 AM] Attorney Michael Grant: Understood. Well submit by the deadline. [11:01 AM] Attorney Lisa Chen: Agreed. [11:02 AM] Arbitrator Hon. Rebecca Lawson: The next hearing is scheduled for October 22 2025 at 1030 AM Eastern Time. Please ensure your witnesses are available for cross examination. [4:45 PM] Arbitrator Hon. Rebecca Lawson: This session is adjourned. Thank you everyone.","max_tokens":2000,"language":"en"}'


}

function stop_docker() {
    cd $WORKPATH/comps/arb_post_hearing_assistant/deployment/docker_compose
    docker compose -f arb_post_hearing_assistant-compose.yaml down --remove-orphans
}

function main() {

    stop_docker

    build_docker_images

    trap stop_docker EXIT

    echo "Test normal env ..."
    start_service
    validate_microservices
    stop_docker

    echo y | docker system prune

}

main
