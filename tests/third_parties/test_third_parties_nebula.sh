#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"

function deploy_and_start_service() {
    cd $WORKPATH/comps/third_parties/nebula/deployment/kubernetes
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.9.1/cert-manager.yaml
    kubectl create namespace nebula-operator-system
    helm repo add nebula-operator https://vesoft-inc.github.io/nebula-operator/charts
    helm repo update
    helm install nebula-operator nebula-operator/nebula-operator --namespace=nebula-operator-system --version=1.1.0

    kubectl create -f community_edition.yaml

    sleep 60s
}

function validate_database() {
    cluster_ip=$(kubectl get service -l app.kubernetes.io/cluster=nebula | awk '/nebula-graphd-svc/ {print $3}')

    # test create space
    echo "[ test create ] creating space.."
    query="CREATE SPACE my_space(partition_num=10, replica_factor=1);"

    create_response=$(kubectl run -ti --image vesoft/nebula-console --restart=Never -- nebula-console -addr "$cluster_ip" -port 9669 -u root -p vesoft -e "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test create ] create space succeed"
        echo $create_response >> ${LOG_PATH}/nebulagraph_create_space.log
    else
        echo "[ test create ] create space failed"
        echo $create_response >> ${LOG_PATH}/nebulagraph_create_space.log
        exit 1
    fi

    # test insert data
    echo "[ test insert ] inserting data.."
    query="USE my_space; CREATE TAG person(name string, age int); INSERT VERTEX person(name, age) VALUES 'person1':('Alice', 30); INSERT VERTEX person(name, age) VALUES 'person2':('Bob', 25);"
   
    insert_response=$(kubectl run -ti --image vesoft/nebula-console --restart=Never -- nebula-console -addr "$cluster_ip" -port 9669 -u root -p vesoft -e "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test insert ] insert data succeed"
        echo $insert_response >> ${LOG_PATH}/nebulagraph_insert_data.log
    else
        echo "[ test insert ] insert data failed"
        echo $insert_response >> ${LOG_PATH}/nebulagraph_insert_data.log
        exit 1
    fi

 
    # test search data
    echo "[ test search ] searching data.."
    query="USE my_space; MATCH (p:person) RETURN p;"

    search_response=$(kubectl run -ti --image vesoft/nebula-console --restart=Never -- nebula-console -addr "$cluster_ip" -port 9669 -u root -p vesoft -e "$query" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "[ test search ] search data succeed"
        echo $search_response >> ${LOG_PATH}/nebulagraph_search_data.log
    else
        echo "[ test search ] search data failed"
        echo $search_response >> ${LOG_PATH}/nebulagraph_search_data.log
        exit 1
    fi
}

function stop_service() {
    cd $WORKPATH/comps/third_parties/nebula/deployment/kubernetes
    kubectl delete -f community_edition.yaml
    helm uninstall nebula-operator --namespace nebula-operator-system
    kubectl delete crd nebulaclusters.apps.nebula-graph.io
    kubectl delete namespace nebula-operator-system
}

function main() {

    deploy_and_start_service

    validate_database

    stop_service

}

main
