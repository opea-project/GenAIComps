#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# service: service path name, like 'agent_langchain', 'asr_whisper'
# hardware: 'intel_cpu', 'intel_hpu', ...

set -xe
cd $WORKSPACE
changed_files_full=$changed_files_full
run_matrix="{\"include\":["

# add test services when comps code change
function find_test_1() {
    local pre_service_path=$1
    local n=$2
    local all_service=$3

    if [ $n -eq 3 ]; then
        pre_service_path=$pre_service_path/src
        n=$((n+1))
    fi

    common_file_change=$(printf '%s\n' "${changed_files[@]}"| grep ${pre_service_path} | cut -d'/' -f$n | grep -E '*.py' | grep -vE '__init__.py|version.py' | sort -u) || true
    if [ "$common_file_change" ] || [ "$all_service" = "true" ]; then
        # if common files changed, run all services
        services=$(ls ${pre_service_path} | cut -d'/' -f$n | grep -vE '*.md|*.py|*.sh|*.yaml|*.yml|*.pdf' | sort -u) || true
        all_service="true"
    else
        # if specific service files changed, only run the specific service
        services=$(printf '%s\n' "${changed_files[@]}"| grep ${pre_service_path} | cut -d'/' -f$n | grep -vE '*.py|*.sh|*.yaml|*.yml|*.pdf' | sort -u) || true
    fi

    for service in ${services}; do
        service_path=$pre_service_path/$service
        if [[ $(ls ${service_path} | grep -E "Dockerfile*") ]]; then
            service_name=$(echo $service_path | sed 's:/src/:/:' | tr '/' '_' | cut -c7-) # comps/retrievers/src/redis/langchain -> retrievers_redis_langchain
            default_service_script_path=$(find ./tests -type f -name test_${service_name}.sh) || true
            if [ "$default_service_script_path" ]; then
                run_matrix="${run_matrix}{\"service\":\"${service_name}\",\"hardware\":\"intel_cpu\"},"
            fi
            other_service_script_path=$(find ./tests -type f -name test_${service_name}_on_*.sh) || true
            for script in ${other_service_script_path}; do
                _service=$(echo $script | cut -d'/' -f4 | cut -d'.' -f1 | cut -c6-)
                hardware=${_service#*_on_}
                run_matrix="${run_matrix}{\"service\":\"${_service}\",\"hardware\":\"${hardware}\"},"
            done
        else
            find_test_1 $service_path $((n+1)) $all_service
        fi
    done
}

# add test case when test scripts code change
function find_test_2() {
    test_files=$(printf '%s\n' "${changed_files[@]}" | grep -E "*.sh") || true
    for test_file in ${test_files}; do
        _service=$(echo $test_file | cut -d'/' -f3 | cut -d'.' -f1 | cut -c6-)
        if [ $(echo ${_service} | grep -c "_on_") == 0 ]; then
            service=${_service}
            hardware="intel_cpu"
        else
            hardware=${_service#*_on_}
        fi
        if [[ $(echo ${run_matrix} | grep -c "{\"service\":\"${_service}\",\"hardware\":\"${hardware}\"},") == 0 ]]; then
            run_matrix="${run_matrix}{\"service\":\"${_service}\",\"hardware\":\"${hardware}\"},"
        fi
    done
}

function main() {

    changed_files=$(printf '%s\n' "${changed_files_full[@]}" | grep 'comps/.*/src/.*' | grep -vE '*.md|comps/cores|comps/3rd_parties') || true
    find_test_1 "comps" 2 false
    sleep 1s
    echo "===========finish find_test_1============"

    changed_files=$(printf '%s\n' "${changed_files_full[@]}" | grep 'tests/' | grep -vE '*.md|*.txt|tests/cores') || true
    find_test_2
    sleep 1s
    echo "===========finish find_test_2============"

    run_matrix=$run_matrix"]}"
    echo "run_matrix=${run_matrix}"
    echo "run_matrix=${run_matrix}" >> $GITHUB_OUTPUT
}

main
