#!/bin/bash

# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

source /GenAIEval/.github/workflows/scripts/change_color
export COVERAGE_RCFILE="/GenAIEval/.github/workflows/scripts/unittest/coveragerc"
LOG_DIR=/GenAIEval/log_dir
mkdir -p ${LOG_DIR}
# get parameters
PATTERN='[-a-zA-Z0-9_]*='
PERF_STABLE_CHECK=true

for i in "$@"; do
    case $i in
        --test_name=*)
            test_name=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

function pytest() {
    local coverage_log_dir="${LOG_DIR}/$1"
    mkdir -p ${coverage_log_dir}
    ut_log_name="${LOG_DIR}/unit_test_$1.log"
    export GLOG_minloglevel=2

    genaieval_path=$(python -c 'import GenAIEval; import os; print(os.path.dirname(GenAIEval.__file__))')
    find . -name "test*.py" | sed 's,\.\/,coverage run --source='"${genaieval_path}"' --append ,g' | sed 's/$/ --verbose/' >run.sh
    coverage erase

    # run UT
    $BOLD_YELLOW && echo "cat run.sh..." && $RESET
    cat run.sh | tee ${ut_log_name}
    $BOLD_YELLOW && echo "------UT start-------" && $RESET
    bash run.sh 2>&1 | tee -a ${ut_log_name}
    $BOLD_YELLOW && echo "------UT end -------" && $RESET

    # run coverage report
    coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log_dir}/coverage.log
    coverage html -d ${coverage_log_dir}/htmlcov --rcfile=${COVERAGE_RCFILE}
    coverage xml -o ${coverage_log_dir}/coverage.xml --rcfile=${COVERAGE_RCFILE}

    # check UT status
    if [ $(grep -c "FAILED" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in UT, please search [FAILED]..." && $RESET
        exit 1
    fi
    if [ $(grep -c "ModuleNotFoundError:" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in UT, please search [ModuleNotFoundError:]..." && $RESET
        exit 1
    fi
    if [ $(grep -c "core dumped" ${ut_log_name}) != 0 ]; then
        $BOLD_RED && echo "Find errors in UT, please search [core dumped]..." && $RESET
        exit 1
    fi
    if [ $(grep -c "OK" ${ut_log_name}) == 0 ]; then
        $BOLD_RED && echo "No pass case found, please check the output..." && $RESET
        exit 1
    fi
    if [ $(grep -c "==ERROR:" ${ut_log_name}) != 0 ]; then
       $BOLD_RED && echo "ERROR found in UT, please check the output..." && $RESET
        exit 1
    fi
    if [ $(grep -c "Segmentation fault" ${ut_log_name}) != 0 ]; then
       $BOLD_RED && echo "Segmentation Fault found in UT, please check the output..." && $RESET
        exit 1
    fi
    if [ $(grep -c "ImportError:" ${ut_log_name}) != 0 ]; then
       $BOLD_RED && echo "ImportError found in UT, please check the output..." && $RESET
        exit 1
    fi
    $BOLD_GREEN && echo "UT finished successfully! " && $RESET
}

function main() {
    cd /GenAIEval/tests || exit 1
    if [ -f "requirements.txt" ]; then
        python -m pip install --default-timeout=100 -r requirements.txt
        pip list
    else
        echo "Not found requirements.txt file."
    fi
    pip install coverage
    pip install pytest
    echo "test on ${test_name}"
    if [[ $test_name == "PR-test" ]]; then
        pytest "pr"
    elif [[ $test_name == "baseline" ]]; then
        pytest "base"
    fi
}

main
