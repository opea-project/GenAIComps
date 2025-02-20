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

    sleep 5s
}

function validate_microservice() {
    # validate neo4j-apoc
    validate_service \
        "${ip_address}:${NEO4J_PORT1}" \
        "200 OK" \
        "neo4j-apoc" \
        "neo4j-apoc" \
        ""
    sleep 1m  # retrieval can't curl as expected, try to wait for more time
    # tgi for llm service
    validate_service \
        "${ip_address}:${LLM_ENDPOINT_PORT}/generate" \
        "generated_text" \
        "tgi-gaudi-service" \
        "tgi-gaudi-server" \
        '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}'

    # test /v1/dataprep graph extraction
    echo "Like many companies in the O&G sector, the stock of Chevron (NYSE:CVX) has declined about 10% over the past 90-days despite the fact that Q2 consensus earnings estimates have risen sharply (~25%) during that same time frame. Over the years, Chevron has kept a very strong balance sheet. FirstEnergy (NYSE:FE – Get Rating) posted its earnings results on Tuesday. The utilities provider reported $0.53 earnings per share for the quarter, topping the consensus estimate of $0.52 by $0.01, RTT News reports. FirstEnergy had a net margin of 10.85% and a return on equity of 17.17%. The Dáil was almost suspended on Thursday afternoon after Sinn Féin TD John Brady walked across the chamber and placed an on-call pager in front of the Minister for Housing Darragh O’Brien during a debate on retained firefighters. Mr O’Brien said Mr Brady had taken part in an act of theatre that was obviously choreographed.Around 2,000 retained firefighters around the country staged a second day of industrial action on Tuesday and are due to start all out-strike action from next Tuesday. The mostly part-time workers, who keep the services going outside of Ireland’s larger urban centres, are taking industrial action in a dispute over pay and working conditions. Speaking in the Dáil, Sinn Féin deputy leader Pearse Doherty said firefighters had marched on Leinster House today and were very angry at the fact the Government will not intervene. Reintroduction of tax relief on mortgages needs to be considered, O’Brien says. Martin withdraws comment after saying People Before Profit would ‘put the jackboot on people’ Taoiseach ‘propagated fears’ farmers forced to rewet land due to nature restoration law – Cairns An intervention is required now. I’m asking you to make an improved offer in relation to pay for retained firefighters, Mr Doherty told the housing minister.I’m also asking you, and challenging you, to go outside after this Order of Business and meet with the firefighters because they are just fed up to the hilt in relation to what you said.Some of them have handed in their pagers to members of the Opposition and have challenged you to wear the pager for the next number of weeks, put up with an €8,600 retainer and not leave your community for the two and a half kilometres and see how you can stand over those type of pay and conditions. At this point, Mr Brady got up from his seat, walked across the chamber and placed the pager on the desk in front of Mr O’Brien. Ceann Comhairle Seán Ó Fearghaíl said the Sinn Féin TD was completely out of order and told him not to carry out a charade in this House, adding it was absolutely outrageous behaviour and not to be encouraged.Mr O’Brien said Mr Brady had engaged in an act of theatre here today which was obviously choreographed and was then interrupted with shouts from the Opposition benches. Mr Ó Fearghaíl said he would suspend the House if this racket continues.Mr O’Brien later said he said he was confident the dispute could be resolved and he had immense regard for firefighters. The minister said he would encourage the unions to re-engage with the State’s industrial relations process while also accusing Sinn Féin of using the issue for their own political gain." > $LOG_PATH/dataprep_file.txt
    validate_service \
        "http://${ip_address}:${DATAPREP_PORT}/v1/dataprep/ingest" \
        "Data preparation succeeded" \
        "extract_graph_neo4j" \
        "dataprep-neo4j-llamaindex"

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
