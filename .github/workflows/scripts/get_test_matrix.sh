#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# service: service path name, like 'agent_langchain', 'asr_whisper'
# hardware: 'intel_cpu', 'intel_hpu', ...

set -e
cd $WORKSPACE
changed_files_full=$changed_files_full
run_matrix="{\"include\":["


function find_test_1() {
    local pre_service_path=$1
    local n=$2
    local all_service=$3

    common_file_change=$(printf '%s\n' "${changed_files[@]}"| grep ${pre_service_path} | cut -d'/' -f$n | grep -E '\.py' | grep -vE '__init__.py|version.py' | sort -u) || true
    if [ "$common_file_change" ] || [ "$all_service" = "true" ]; then
        # if common files changed, run all services
        services=$(ls ${pre_service_path} | cut -d'/' -f$n | grep -vE '\.md|\.py|\.sh|\.yaml|\.yml|\.pdf' | sort -u) || true
        all_service="true"
    else
        # if specific service files changed, only run the specific service
        services=$(printf '%s\n' "${changed_files[@]}"| grep ${pre_service_path} | cut -d'/' -f$n | grep -vE '\.py|\.sh|\.yaml|\.yml|\.pdf' | sort -u) || true
    fi

    for service in ${services}; do
        service_path=$pre_service_path/$service
        if [[ $(ls ${service_path} | grep -E "Dockerfile*") ]]; then
            if [[ $(ls ${service_path} | grep "integrations") ]]; then
                # new org with `src` and `integrations` folder
                run_all_interation="false"
                service_name=$(echo $service_path | sed 's:/src::' | tr '/' '_' | cut -c7-) # comps/retrievers/src/redis/langchain -> retrievers_redis_langchain
                common_file_change_insight=$(printf '%s\n' "${changed_files[@]}"| grep ${service_path} | grep -vE 'integrations' | sort -u) || true
                if [ "$common_file_change_insight" ]; then
                    # if common file changed, run all integrations
                    run_all_interation="true"
                fi
                if [ "$run_all_interation" = "false" ]; then
                    changed_integrations=$(printf '%s\n' "${changed_files[@]}"| grep ${service_path} | grep -E 'integrations' | cut -d'/' -f$((n+2)) | cut -d'.' -f1 | sort -u)  || true
                    for integration in ${changed_integrations}; do
                        # Accurate matching test scripts
                        # find_test=$(find ./tests -type f \( -name test_${service_name}_${integrations}.sh -o -name test_${service_name}_${integrations}_on_*.sh \)) || true
                        # Fuzzy matching test scripts, for example, llms/src/text-generation/integrations/opea.py match several tests.
                        find_test=$(find ./tests -type f -name test_${service_name}_${integration}*.sh) || true
                        if [ "$find_test" ]; then
                            fill_in_matrix "$find_test"
                        else
                            run_all_interation="true"
                            break
                        fi
                    done
                fi
                if [ "$run_all_interation" = "true" ]; then
                    find_test=$(find ./tests -type f -name test_${service_name}*.sh) || true
                    if [ "$find_test" ]; then
                        fill_in_matrix "$find_test"
                    fi
                fi
            elif [[ $(echo ${service_path} | grep "third_parties") ]]; then
                 # new org with `src` and `third_parties` folder
                service_name=$(echo $service_path | sed 's:/src::' | tr '/' '_' | cut -c7-) # comps/third_parties/vllm/src -> third_parties_vllm
                find_test=$(find ./tests -type f -name test_${service_name}*.sh) || true
                if [ "$find_test" ]; then
                    fill_in_matrix "$find_test"
                fi
            else
                # old org without 'src' folder
                service_name=$(echo $service_path | tr '/' '_' | cut -c7-) # comps/retrievers/redis/langchain -> retrievers_redis_langchain
                find_test=$(find ./tests -type f -name test_${service_name}*.sh) || true
                if [ "$find_test" ]; then
                    fill_in_matrix "$find_test"
                fi
            fi
        else
            find_test_1 $service_path $((n+1)) $all_service
        fi
    done
}

function fill_in_matrix() {
    find_test=$1
    for test in ${find_test}; do
        _service=$(echo $test | cut -d'/' -f4 | cut -d'.' -f1 | cut -c6-)
        _fill_in_matrix $_service
    done
}

function _fill_in_matrix() {
    _service=$1
    if [ $(echo ${_service} | grep -c "_on_") == 0 ]; then
        service=${_service}
        hardware="intel_cpu"
    else
        hardware=${_service#*_on_}
    fi
    echo "service=${_service}, hardware=${hardware}"
    if [[ $(echo ${run_matrix} | grep -c "{\"service\":\"${_service}\",\"hardware\":\"${hardware}\"},") == 0 ]]; then
        run_matrix="${run_matrix}{\"service\":\"${_service}\",\"hardware\":\"${hardware}\"},"
        echo "------------------ add one service ------------------"
    fi
    sleep 1s
}


function find_test_2() {
    test_files=$(printf '%s\n' "${changed_files[@]}" | grep -E "\.sh" | grep -E "test_") || true
    for test_file in ${test_files}; do
        if [ -f $test_file ]; then
            _service=$(echo $test_file | cut -d'/' -f3 | grep -E "\.sh" | cut -d'.' -f1 | cut -c6-)
            if [ -n "${_service}" ]; then
                _fill_in_matrix $_service
            fi
        fi
    done
}


function find_test_3() {
    yaml_files=${changed_files}
    for yaml_file in ${yaml_files}; do
        if [ -f $yaml_file ]; then
            _service=${yaml_file#comps/}
            _service=${_service%/deployment/*}
            _service=${_service//\//_}
            yaml_name=$(basename $yaml_file)
            if [ "$yaml_name" != "compose.yaml" ]; then
                _domain=${yaml_name%.yaml}
                _domain=${_domain#compose_}
                _service=${_service}_${_domain}
            fi
            find_test=$(find ./tests -type f -name test_${_service}*.sh) || true
            if [ "$find_test" ]; then
                fill_in_matrix "$find_test"
            fi
        fi
    done
}


function main() {

    # add test services when comps code change
    changed_files=$(printf '%s\n' "${changed_files_full[@]}" | grep 'comps/' | grep -vE '\.md|comps/cores|deployment|\.yaml') || true
    echo "===========start find_test_1============"
    echo "changed_files=${changed_files}"
    find_test_1 "comps" 2 false
    sleep 1s
    echo "run_matrix=${run_matrix}"
    echo "===========finish find_test_1============"

    # add test case when test scripts code change
    changed_files=$(printf '%s\n' "${changed_files_full[@]}" | grep 'tests/' | grep -vE '\.md|\.txt|tests/cores') || true
    echo "===========start find_test_2============"
    echo "changed_files=${changed_files}"
    find_test_2
    sleep 1s
    echo "run_matrix=${run_matrix}"
    echo "===========finish find_test_2============"

    # add test case when docker-compose code change
    changed_files=$(printf '%s\n' "${changed_files_full[@]}" | grep 'deployment/docker_compose/compose' | grep '.yaml') || true
    echo "===========start find_test_3============"
    echo "changed_files=${changed_files}"
    find_test_3
    sleep 1s
    echo "run_matrix=${run_matrix}"
    echo "===========finish find_test_3============"

    run_matrix=$run_matrix"]}"
    echo "run_matrix=${run_matrix}"
    echo "run_matrix=${run_matrix}" >> $GITHUB_OUTPUT

    if [[ $(echo "$run_matrix" | grep -c "service") != 0 ]]; then
        is_empty="false"
    fi
    echo "is_empty=${is_empty}"
    echo "is_empty=${is_empty}" >> $GITHUB_OUTPUT
}

main
