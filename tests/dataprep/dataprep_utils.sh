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
      delete|get)
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

# validate_ingest <service fqdn> <port>
function ingest_doc() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.doc" $@
}

function ingest_docx() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.docx" $@
}

function ingest_pdf() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.pdf" $@
}

function ingest_pptx() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.pptx" $@
}

function ingest_txt() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.txt" $@
}

function ingest_xlsx() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F "files=@${SCRIPT_DIR}/ingest_dataprep.xlsx" $@
}

function ingest_external_link() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port ingest -F 'link_list=["https://www.ces.tech/"]' $@
}

function delete_all() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port delete -d '{"file_path":"all"}' $@
}

function delete_single() {
    local fqdn=$1
    local port=$2
    shift 3
    _invoke_curl $fqdn $port delete -d '{"file_path":"ingest_dataprep.txt"}' $@
}

function get_all() {
    local fqdn=$1
    local port=$2
    shift 2
    _invoke_curl $fqdn $port get $@
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
