#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')
DATAPREP_PORT="11108"
TEI_EMBEDDER_PORT="10221"
export TAG="comps"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ${SCRIPT_DIR}/dataprep_utils.sh

function build_docker_images() {
    cd $WORKPATH
    echo $(pwd)
    docker build -t opea/dataprep:${TAG} --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/dataprep built fail"
        exit 1
    else
        echo "opea/dataprep built successful"
    fi
}

function build_vllm_docker_images() {
    echo "Building the vllm docker images"
    cd $WORKPATH
    echo $WORKPATH
    if [ ! -d "./vllm-fork" ]; then
        git clone https://github.com/HabanaAI/vllm-fork.git
    fi
    cd ./vllm-fork
    VLLM_VER=$(git describe --tags "$(git rev-list --tags --max-count=1)")
    echo "Check out vLLM tag ${VLLM_VER}"
    git checkout ${VLLM_VER} &> /dev/null

    # cd $WORKDIR/vllm-fork

    docker build --no-cache -f Dockerfile.hpu -t opea/vllm-gaudi:comps --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
    if [ $? -ne 0 ]; then
        echo "opea/vllm-gaudi:comps failed"
        exit 1
    else
        echo "opea/vllm-gaudi:comps successful"
    fi
}

function start_vllm_service_70B() {
    echo "token is ${HF_TOKEN}"
    model="meta-llama/Llama-3.3-70B-Instruct"
    vllm_port=8086
    export HF_CACHE_DIR=${model_cache:-./data}
    vllm_volume=$HF_CACHE_DIR

    echo "start vllm gaudi service"
    echo "**************model is $model**************"
    docker run -d --runtime=habana --rm --name "test-comps-vllm-gaudi-service" -e HABANA_VISIBLE_DEVICES=0,1,2,3 -p $vllm_port:8000 -v $vllm_volume:/data -e HF_TOKEN=$HF_TOKEN -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN -e HF_HOME=/data -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e PT_HPU_ENABLE_LAZY_COLLECTIVES=true -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e VLLM_SKIP_WARMUP=true --cap-add=sys_nice --ipc=host opea/vllm-gaudi:comps --model ${model} --max-seq-len-to-capture 16384 --tensor-parallel-size 4
    sleep 5s
    echo "Waiting vllm gaudi ready"
    n=0
    until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
        docker logs test-comps-vllm-gaudi-service &> ${LOG_PATH}/vllm-gaudi-service.log
        n=$((n+1))
        if grep -q "Uvicorn running on" ${LOG_PATH}/vllm-gaudi-service.log; then
            break
        fi
        if grep -q "No such container" ${LOG_PATH}/vllm-gaudi-service.log; then
            echo "container test-comps-vllm-gaudi-service not found"
            exit 1
        fi
        sleep 5s
    done
    sleep 5s
    echo "Service started successfully"
}


function stop_vllm_service() {
    cid=$(docker ps -aq --filter "name=test-comps-vllm-gaudi-service")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function start_service() {

    export host_ip=${ip_address}
    export DATAPREP_PORT="11108"
    export TEI_EMBEDDER_PORT="10221"
    export REDIS_URL_VECTOR="redis://${ip_address}:6379"
    export REDIS_URL_KV="redis://${ip_address}:6380"
    export LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct"
    export LLM_ENDPOINT="http://${ip_address}:8086"
    export DATAPREP_COMPONENT_NAME="OPEA_DATAPREP_REDIS_FINANCE"
    export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${TEI_EMBEDDER_PORT}"

    service_name="redis-vector-db redis-kv-store tei-embedding-serving dataprep-redis-finance"
    cd $WORKPATH/comps/dataprep/deployment/docker_compose/
    docker compose up ${service_name} -d
    sleep 1m
}

function validate() {
    local CONTENT="$1"
    local EXPECTED_RESULT="$2"
    local SERVICE_NAME="$3"
    echo "EXPECTED_RESULT: $EXPECTED_RESULT"
    echo "Content: $CONTENT"
    if echo "$CONTENT" | grep -q "$EXPECTED_RESULT"; then
        echo "[ $SERVICE_NAME ] Content is as expected: $CONTENT"
        echo 0
    else
        echo "[ $SERVICE_NAME ] Content does not match the expected result: $CONTENT"
        echo 1
    fi
}

function validate_microservice() {
    # test /v1/dataprep/ingest
    echo "=========== Test ingest ==========="
    local CONTENT=$(python $WORKPATH/tests/dataprep/test_redis_finance.py --port $DATAPREP_PORT --test_option ingest)
    local EXIT_CODE=$(validate "$CONTENT" "200" "dataprep-redis-finance")
    echo "$EXIT_CODE"
    local EXIT_CODE="${EXIT_CODE:0-1}"
    if [ "$EXIT_CODE" == "1" ]; then
        docker logs dataprep-redis-server-finance &> ${LOG_PATH}/dataprep_ingest.log
        exit 1
    fi

    # test /v1/dataprep/get
    echo "=========== Test get ==========="
    local CONTENT=$(python $WORKPATH/tests/dataprep/test_redis_finance.py --port $DATAPREP_PORT --test_option get)
    local EXIT_CODE=$(validate "$CONTENT" "Request successful" "dataprep-redis-finance")
    echo "$EXIT_CODE"
    local EXIT_CODE="${EXIT_CODE:0-1}"
    if [ "$EXIT_CODE" == "1" ]; then
        docker logs dataprep-redis-server-finance &> ${LOG_PATH}/dataprep_get.log
        exit 1
    fi

    # test /v1/dataprep/delete
    echo "=========== Test delete ==========="
    local CONTENT=$(python $WORKPATH/tests/dataprep/test_redis_finance.py --port $DATAPREP_PORT --test_option delete)
    local EXIT_CODE=$(validate "$CONTENT" "Request successful" "dataprep-redis-finance")
    echo "$EXIT_CODE"
    local EXIT_CODE="${EXIT_CODE:0-1}"
    if [ "$EXIT_CODE" == "1" ]; then
        docker logs dataprep-redis-server-finance &> ${LOG_PATH}/dataprep_delete.log
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=dataprep-redis-server*" --filter "name=redis-vector-*" --filter "name=redis-kv-store" --filter "name=tei-embedding-*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker
    stop_vllm_service

    build_docker_images
    start_service

    build_vllm_docker_images
    start_vllm_service_70B

    validate_microservice

    stop_vllm_service
    stop_docker
    echo y | docker system prune

}

main
