#!/usr/bin/env bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# call_curl <url> <http_header> <remaining params>
function call_curl() {
    local url=$1
    local header=$2
    shift 2
    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -H "$header" "${url}" $@)
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')
}

# _invoke_curl <fqdn> <port> <action> <remaining params passed to curl ...>
function _invoke_curl() {
    local url="http://$1:$2/v1/dataprep/$3"
    local action=$3
    shift 3
    local header=""
    case $action in
      ingest)
        header='Content-Type: multipart/form-data'
        ;;
      delete|get|indices)
        header='Content-Type: application/json'
	;;
      *)
        echo "Error: Unsupported dataprep action $action!"
        exit 1
	;;
    esac

    HTTP_RESPONSE=$(curl --silent --write-out "HTTPSTATUS:%{http_code}" -X POST -H "$header" "${url}" $@)
    HTTP_STATUS=$(echo $HTTP_RESPONSE | tr -d '\n' | sed -e 's/.*HTTPSTATUS://')
    RESPONSE_BODY=$(echo $HTTP_RESPONSE | sed -e 's/HTTPSTATUS\:.*//g')
}

#
function _add_db_params() {
    local db=$1
    if [[ "$db" == "redis" ]]; then
        echo '-F index_name=test'
    fi
}

# validate_ingest <service fqdn> <port>
function ingest_doc() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.doc" $extra_args $@
}

function ingest_docx() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.docx" $extra_args $@
}

function ingest_pdf() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.pdf" $extra_args $@
}

function ingest_ppt() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.ppt" $extra_args $@
}

function ingest_pptx() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.pptx" $extra_args $@
}

function ingest_txt() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.txt" $extra_args $@
}

function ingest_xlsx() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.xlsx" $extra_args $@
}

function ingest_external_link() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port ingest -F 'link_list=["https://www.ces.tech/"]' $extra_args $@
}

function delete_all() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port delete -d '{"file_path":"all"}' $@
}

function delete_all_in_index() {
    local fqdn=$1
    local port=$2
    local index_name=$3
    shift 3
    _invoke_curl $fqdn $port delete -d '{"file_path":"all","index_name":"'${index_name}'"}' $@
}

function delete_item_in_index() {
    local fqdn=$1
    local port=$2
    local index_name=$3
    local item=$4
    shift 4
    _invoke_curl $fqdn $port delete -d '{"file_path":"'${item}'","index_name":"'${index_name}'"}' $@
}

function delete_single() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port delete -d '{"file_path":"ingest_dataprep.txt"}' $@
}

function get_all() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port get $@
}

function get_all_in_index() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port get -d '{"index_name":"all"}' $@
}

function get_index() {
    local fqdn=$1
    local port=$2
    local index_name=$3
    shift 3
    _invoke_curl $fqdn $port get -d '{"index_name":"'${index_name}'"}' $@
}

function ingest_txt_with_index_name() {
    local fqdn=$1
    local port=$2
    local index_name=$3
    shift 3
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.txt" -F "index_name=${index_name}" $@
}

function indices() {
    local fqdn=$1
    local port=$2
    local db=$3
    shift 3
    local extra_args=$(_add_db_params "$db")
    _invoke_curl $fqdn $port indices $@
}

function check_result() {
    local service_name=$1
    local expected_response=$2
    local container_name=$3
    local logfile=$4
    local http_status="${5:-200}"

    if [ "$HTTP_STATUS" -ne ${http_status} ]; then
        echo "[ $service_name ] HTTP status is not ${http_status}. Received status was $HTTP_STATUS"
        docker logs $container_name >> $logfile
        exit 1
    else
        echo "[ $service_name ] HTTP status is ${http_status}. Checking content..."
    fi

     # check response body
    if [[ "$RESPONSE_BODY" != *${expected_response}* ]]; then
        echo "[ $service_name ] Content does not match the expected result: $RESPONSE_BODY"
        docker logs $container_name >> $logfile
        exit 1
    else
        echo "[ $service_name ] Content is as expected."
    fi
}

function check_healthy() {
    local container_name=$1
    local retries=30
    local count=0

    echo "Waiting for $container_name to become healthy..."

    while [ $count -lt $retries ]; do
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null)
        if [ "$status" == "healthy" ]; then
            echo "$container_name is healthy!"
            return 0
        fi
        echo "  → $container_name status: $status ($count/$retries)"
        sleep 5
        ((count++))
    done

    echo "$container_name did not become healthy in time."
    return 1
}
