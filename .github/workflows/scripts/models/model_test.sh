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
source /GenAIEval/.github/workflows/script/change_color.sh

# get parameters
PATTERN='[-a-zA-Z0-9_]*='
PERF_STABLE_CHECK=true
for i in "$@"; do
    case $i in
        --device=*)
            device=`echo $i | sed "s/${PATTERN}//"`;;
        --model=*)
            model=`echo $i | sed "s/${PATTERN}//"`;;
        --task=*)
            task=`echo $i | sed "s/${PATTERN}//"`;;
        *)
            echo "Parameter $i not recognized."; exit 1;;
    esac
done

log_dir="/GenAIEval/${device}/${model}"
mkdir -p ${log_dir}

$BOLD_YELLOW && echo "-------- evaluation start --------" && $RESET

main() {
    #prepare
    run_benchmark
}

function prepare() {
    ## prepare env
    working_dir="/GenAIEval"
    cd ${working_dir}
    echo "Working in ${working_dir}"
    echo -e "\nInstalling model requirements..."
    if [ -f "requirements.txt" ]; then
        python -m pip install -r requirements.txt
        pip list
    else
        echo "Not found requirements.txt file."
    fi
}

function run_benchmark() {
    cd ${working_dir}
    pip install --upgrade-strategy eager optimum[habana]
    overall_log="${log_dir}/${device}-${model}-${task}.log"
    python main.py \
        --model hf \
        --model_args pretrained=${model} \
        --tasks ${task} \
        --device ${device} \
        --batch_size 8
        2>&1 | tee ${overall_log}

    status=$?
    if [ ${status} != 0 ]; then
        echo "Evaluation process returned non-zero exit code."
        exit 1
    fi
}

main
