#!/usr/bin/env bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

IMAGE_REPO=${IMAGE_REPO:-"opea"}
export REGISTRY=${IMAGE_REPO}
export TAG="comps"
echo "REGISTRY=IMAGE_REPO=${IMAGE_REPO}"
echo "TAG=${TAG}"

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
export host_ip=$(hostname -I | awk '{print $1}')
service_name="retriever-neo4j"

function build_docker_images() {
    cd $WORKPATH
    echo "current dir: $PWD"
    docker build --no-cache -t ${REGISTRY:-opea}/retriever:${TAG:-latest} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/retriever built fail"
        exit 1
    else
        echo "opea/retriever built successful"
    fi

    docker build --no-cache -t opea/dataprep-neo4j-llamaindex:comps --build-arg no_proxy=$no_proxy --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep-neo4j-llamaindex built fail"
        exit 1
    else
        echo "opea/dataprep-neo4j-llamaindex built successful"
    fi

    docker pull ghcr.io/huggingface/tgi-gaudi:2.3.1
    docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.5
}

function start_service() {
    export TEI_EMBEDDER_PORT=11633
    export LLM_ENDPOINT_PORT=11634
    export RETRIEVER_PORT=11635
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export DATA_PATH="/data2/cache"
    export MAX_INPUT_TOKENS=4096
    export MAX_TOTAL_TOKENS=8192
    export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
    export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
    export TGI_LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
    export DATAPREP_SERVICE_ENDPOINT="http://${host_ip}:6004"
    export NEO4J_PORT1=7474   # 11631
    export NEO4J_PORT2=7687   # 11632
    export NEO4J_URI="bolt://${host_ip}:${NEO4J_PORT2}"
    export NEO4J_URL="bolt://${host_ip}:${NEO4J_PORT2}"
    export NEO4J_USERNAME="neo4j"
    export NEO4J_PASSWORD="neo4jtest"
    export no_proxy="localhost,127.0.0.1,"${host_ip}
    export LOGFLAG=True

    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml up ${service_name} -d > ${LOG_PATH}/start_services_with_compose.log
    sleep 1m

    # dataprep neo4j
    # Not testing openai code path since not able to provide key for cicd
    docker run -d --name="test-comps-retrievers-neo4j-llama-index-dataprep" -p 6004:5000 -v ./data:/data --ipc=host -e TGI_LLM_ENDPOINT=$TGI_LLM_ENDPOINT \
        -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e EMBEDDING_MODEL_ID=$EMBEDDING_MODEL_ID -e LLM_MODEL_ID=$LLM_MODEL_ID -e host_ip=$host_ip -e no_proxy=$no_proxy \
        -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e NEO4J_URL="bolt://${host_ip}:${NEO4J_PORT2}" -e NEO4J_USERNAME="neo4j" \
        -e NEO4J_PASSWORD="neo4jtest" -e NEO4J_PORT1=$NEO4J_PORT1 -e NEO4J_PORT2=$NEO4J_PORT2 -e HF_TOKEN=$HF_TOKEN -e MAX_INPUT_TOKENS=$MAX_INPUT_TOKENS -e LOGFLAG=True \
        -e DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_NEO4J_LLAMAINDEX" opea/dataprep-neo4j-llamaindex:comps

    sleep 1m

}

function validate_service() {
    local URL="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_NAME="$3"
    local DOCKER_NAME="$4"
    local INPUT_DATA="$5"

    if [[ $SERVICE_NAME == *"extract_graph_neo4j"* ]]; then
        cd $LOG_PATH
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -F 'files=@./dataprep_file.txt' -H 'Content-Type: multipart/form-data' "$URL")
    elif [[ $SERVICE_NAME == *"neo4j-apoc"* ]]; then
         HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" "$URL")
    else
        HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -d "$INPUT_DATA" -H 'Content-Type: application/json' "$URL")
    fi
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')

    docker logs ${DOCKER_NAME} >> ${LOG_PATH}/${SERVICE_NAME}.log

    # check response status
    if [ "$HTTP_STATUS" -ne "200" ]; then
        echo "[ $SERVICE_NAME ] HTTP status is not 200. Received status was $HTTP_STATUS"
        exit 1
    else
        echo "[ $SERVICE_NAME ] HTTP status is 200. Checking content..."
    fi
    # check response body
    if [[ "$SERVICE_NAME" == "neo4j-apoc" ]]; then
        echo "[ $SERVICE_NAME ] Skipping content check for neo4j-apoc."
    else
        if [[ "$RESPONSE_BODY" != *"$EXPECTED_RESULT"* ]]; then
            echo "[ $SERVICE_NAME ] Content does not match the expected result: $RESPONSE_BODY"
            exit 1
        else
            echo "[ $SERVICE_NAME ] Content is as expected."
        fi
    fi

    sleep 1s
}

function validate_microservice() {
    # validate neo4j-apoc
    validate_service \
        "${host_ip}:${NEO4J_PORT1}" \
        "200 OK" \
        "neo4j-apoc" \
        "neo4j-apoc" \
        ""
    sleep 1m  # retrieval can't curl as expected, try to wait for more time

    # tgi for llm service
    validate_service \
        "${host_ip}:${LLM_ENDPOINT_PORT}/generate" \
        "generated_text" \
        "tgi-gaudi-service" \
        "tgi-gaudi-server" \
        '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}'

    # test /v1/dataprep graph extraction
    echo "The stock of company Chevron has declined about 10% over the past 90-days despite the fact that Q2 consensus earnings estimates have risen sharply (~25%) during that same time frame. Over the years, Chevron has kept a very strong balance sheet. FirstEnergy company posted its earnings results on Tuesday. The utilities provider reported $0.53 earnings per share for the quarter, topping the consensus estimate of $0.52 by $0.01, RTT News reports. FirstEnergy had a net margin of 10.85% and a return on equity of 17.17%. The Dáil was almost suspended on Thursday afternoon after Sinn Féin TD John Brady walked across the chamber and placed an on-call pager in front of the Minister for Housing Darragh O’Brien during a debate on retained firefighters. Darragh O’Brien said John Brady had taken part in an act of theatre that was obviously choreographed. Around 2,000 retained firefighters around the country staged a second day of industrial action on Tuesday and are due to start all out-strike action from next Tuesday. The mostly part-time workers, who keep the services going outside of Ireland’s larger urban centres, are taking industrial action in a dispute over pay and working conditions. Speaking in the Dáil, Sinn Féin deputy leader Pearse Doherty said firefighters had marched on Leinster House today and were very angry at the fact the Government will not intervene. Reintroduction of tax relief on mortgages needs to be considered, Darragh O’Brien says. Martin withdraws comment after saying People Before Profit would ‘put the jackboot on people’ Taoiseach ‘propagated fears’ farmers forced to rewet land due to nature restoration law – Cairns An intervention is required now. I’m asking you to make an improved offer in relation to pay for retained firefighters, Mr Doherty told the housing minister. I’m also asking you, and challenging you, to go outside after this Order of Business and meet with the firefighters because they are just fed up to the hilt in relation to what you said. Some of them have handed in their pagers to members of the Opposition and have challenged you to wear the pager for the next number of weeks, put up with an €8,600 retainer and not leave your community for the two and a half kilometres and see how you can stand over those type of pay and conditions. At this point, John Brady got up from his seat, walked across the chamber and placed the pager on the desk in front of Darragh O’Brien. Ceann Comhairle Seán Ó Fearghaíl said the Sinn Féin TD was completely out of order and told him not to carry out a charade in this House, adding it was absolutely outrageous behaviour and not to be encouraged. Darragh O’Brien said John Brady had engaged in an act of theatre here today which was obviously choreographed and was then interrupted with shouts from the Opposition benches. Mr Ó Fearghaíl said he would suspend the House if this racket continues. Darragh O’Brien later said he was confident the dispute could be resolved and he had immense regard for firefighters. The minister said he would encourage the unions to re-engage with the State’s industrial relations process while also accusing Sinn Féin of using the issue for their own political gain." > $LOG_PATH/dataprep_file.txt
    validate_service \
        "http://${host_ip}:6004/v1/dataprep/ingest" \
        "Data preparation succeeded" \
        "extract_graph_neo4j" \
        "test-comps-retrievers-neo4j-llama-index-dataprep"

    # retrieval microservice
    validate_service \
        "${host_ip}:${RETRIEVER_PORT}/v1/retrieval" \
        "documents" \
        "retriever_community_answers_neo4j" \
        "${service_name}" \
        "{\"messages\": [{\"role\": \"user\",\"content\": \"Who is John Brady and has he had any confrontations?\"}]}"

}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=test-comps-*" --filter "name=neo4j-apoc" --filter "name=tgi-gaudi-server" --filter "name=tei-embedding-serving")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
    cd $WORKPATH/comps/retrievers/deployment/docker_compose
    docker compose -f compose.yaml down  ${service_name} --remove-orphans
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
