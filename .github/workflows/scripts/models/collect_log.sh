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
WORKSPACE="/GenAIEval"
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

output_file="/GenAIEval/${device}/${model}/${device}-${model}-${task}.log"
$BOLD_YELLOW && echo "-------- Collect logs --------" && $RESET

echo "working in"
pwd
if [[ ! -f ${output_file} ]]; then
    echo "${device};${model};${task};;${logfile}" >> ${WORKSPACE}/summary.log
else
    acc=$(grep -Po "Accuracy .* is:\\s+(\\d+(\\.\\d+)?)" ${acc_log_name} | head -n 1 | sed 's/.*://;s/[^0-9.]//g')
    echo "${device};${model};${task};${acc};${logfile}" >> ${WORKSPACE}/summary.log
fi
