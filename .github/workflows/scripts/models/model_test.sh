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

set -eo pipefail
source /GenAIEval/.github/workflows/scripts/change_color

# get parameters
PATTERN='[-a-zA-Z0-9_]*='
PERF_STABLE_CHECK=true
for i in "$@"; do
    case $i in
        --datasets*)
            datasets=`echo $i | sed "s/${PATTERN}//"`;;
        --device=*)
            device=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --tasks=*)
            tasks=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

log_dir="/GenAIEval/${device}/${model}"
mkdir -p ${log_dir}
working_dir=""
$BOLD_YELLOW && echo "-------- evaluation start --------" && $RESET

main() {
    case ${tasks} in
        "text-generation")
            working_dir="/GenAIEval/GenAIEval/evaluation/lm_evaluation_harness/examples";;
        "code-generation")
            working_dir="/GenAIEval/GenAIEval/evaluation/bigcode_evaluation_harness/examples";;
        *)
            echo "Not suppotted task"; exit 1;;
    esac
    run_benchmark
}

function prepare() {
    ## prepare env
    cd ${working_dir}
    echo "Working in ${working_dir}"
    echo -e "\nInstalling model requirements..."
    if [ -f "requirements.txt" ]; then
        python -m pip install -r requirements.txt
        pip list
    else
        echo "Not found requirements.txt file."
    fi
    if [[ ${device} == "hpu" ]]; then
        pip install --upgrade-strategy eager optimum[habana]
    fi
}

function run_benchmark() {
    cd ${working_dir}
    overall_log="${log_dir}/${device}-${model}-${tasks}-${datasets}.log"
    python main.py \
        --model hf \
        --model_args pretrained=${model} \
        --tasks ${datasets} \
        --device ${device} \
        --batch_size 112
        2>&1 | tee ${overall_log}

    status=$?
    if [ ${status} != 0 ]; then
        echo "Evaluation process returned non-zero exit code."
        exit 1
    fi
}

main
