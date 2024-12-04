#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#set -xe

WORKPATH=$(dirname "$PWD")
echo $WORKPATH
LOG_PATH="$WORKPATH/tests"

# WORKDIR is one level up from GenAIComps
WORKDIR=$(dirname "$WORKPATH")
echo $WORKDIR

export ip_address=$(hostname -I | awk '{print $1}')
tgi_port=8085
tgi_volume=$WORKPATH/data
export model=meta-llama/Meta-Llama-3.1-70B-Instruct
export HUGGINGFACEHUB_API_TOKEN=${HF_TOKEN}
export LLM_MODEL_ID="meta-llama/Meta-Llama-3.1-70B-Instruct"
export LLM_ENDPOINT_URL="http://${ip_address}:${tgi_port}"
export temperature=0.01
export max_new_tokens=4096
export TOOLSET_PATH=$WORKPATH/comps/agent/langchain/tools/
echo "TOOLSET_PATH=${TOOLSET_PATH}"
export recursion_limit=15
export db_name=california_schools
export db_path=${WORKDIR}/TAG-Bench/dev_folder/dev_databases/${db_name}/${db_name}.sqlite

# download the test data
function prepare_data() {
    cd $WORKDIR

    echo "Downloading data..."
    git clone https://github.com/TAG-Research/TAG-Bench.git
    cd TAG-Bench/setup
    chmod +x get_dbs.sh
    ./get_dbs.sh

    echo "Split data..."
    cd $WORKPATH/tests/agent/sql_agent_test
    bash run_data_split.sh

    echo "Data preparation done!"
}

# launch tgi-gaudi
function start_tgi_service() {
    echo "token is ${HF_TOKEN}"

    #multi cards
    echo "start tgi gaudi service"
    docker run -d --runtime=habana --name "test-comps-tgi-gaudi-service" -p $tgi_port:80 -v $tgi_volume:/data -e HF_TOKEN=$HF_TOKEN -e HABANA_VISIBLE_DEVICES=0,1,2,3 -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tgi-gaudi:2.0.5 --model-id $model --max-input-tokens 4096 --max-total-tokens 8192 --sharded true --num-shard 4
    sleep 5s
    echo "Waiting tgi gaudi ready"
    n=0
    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
        docker logs test-comps-tgi-gaudi-service &> ${LOG_PATH}/tgi-gaudi-service.log
        n=$((n+1))
        if grep -q Connected ${LOG_PATH}/tgi-gaudi-service.log; then
            break
        fi
        sleep 5s
    done
    sleep 5s
    echo "Service started successfully"
}

# launch the agent
function start_sql_agent_llama_service() {
    echo "Starting sql_agent_llama agent microservice"
    docker compose -f $WORKPATH/tests/agent/sql_agent_llama.yaml up -d
    sleep 5s
    docker logs test-comps-agent-endpoint
    echo "Service started successfully"
}

# run the test
function run_test() {
    echo "Running test..."
    cd $WORKPATH/tests/agent/
    python3 test.py --test-sql-agent
}



