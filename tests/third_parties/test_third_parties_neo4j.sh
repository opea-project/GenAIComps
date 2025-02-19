#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set +e
set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
export STORAGE_CLASS_NAME=$(kubectl get storageclass -o jsonpath='{.items[0].metadata.name}')

if [ -n "$STORAGE_CLASS_NAME" ]; then
    echo "Using StorageClass: $STORAGE_CLASS_NAME"
else
    echo "No StorageClass found."
    exit 1
fi

function deploy_and_start_service() {
    stop_service
    sleep 60s

    cd $WORKPATH/comps/third_parties/neo4j/deployment/kubernetes

    kubectl create namespace neo4j-system
    helm repo add neo4j https://helm.neo4j.com/neo4j && helm repo update

    #helm install -n neo4j-system --version 5.23.0 graph-neo neo4j/neo4j -f cpu.yaml --set storageClassName=$STORAGE_CLASS_NAME
    helm install -n neo4j-system --version 5.23.0 graph-neo neo4j/neo4j --set authEnabled=true --set neo4j.username=neo4j --set neo4j.password=neo4j -f cpu.yaml
    sleep 120s
}

function validate_database() {
    pod_name=$(kubectl get pods -n neo4j-system -l app=neo4j | awk '/graph-neo/ {print $1}')

    if [ -n "$pod_name" ]; then
        echo "Using pod_name: $pod_name"
    else
        echo "No pod_name found."
        exit 1
    fi

    # test query
    echo "[ test query ] querying database.."
    query="MATCH (n) RETURN n LIMIT 5;"

    query_response=$(kubectl exec -it -n neo4j-system "$pod_name" -- cypher-shell -u neo4j -p neo4j "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test query ] query succeed"
        echo $query_response >> ${LOG_PATH}/neo4j_query.log
    else
        echo "[ test query ] query failed"
        echo $query_response >> ${LOG_PATH}/neo4j_query.log
        exit 1
    fi
}

function stop_service() {
    cd $WORKPATH/comps/third_parties/neo4j/deployment/kubernetes
    helm uninstall -n neo4j-system graph-neo && tsleep=5
    neo_pvc=( $(kubectl get pvc -l helm.neo4j.com/instance=graph-neo --no-headers -o custom-columns=CONTAINER:.metadata.name))
    for pvc in "${neo_pvc[@]}"; do
        kubectl delete pvc $pvc
    done
    sleep 30

    kubectl delete namespace neo4j-system
}

function main() {

    deploy_and_start_service

    validate_database

    stop_service

}

main
