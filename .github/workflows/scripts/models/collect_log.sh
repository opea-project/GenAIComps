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
WORKSPACE="/GenAIEval"
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

log_file="/GenAIEval/${device}/${model}/${device}-${model}-${tasks}-${datasets}.log"
$BOLD_YELLOW && echo "-------- Collect logs --------" && $RESET

echo "working in"
pwd
if [[ ! -f ${log_file} ]]; then
    echo "${device};${model};${tasks};${datasets};;${logfile}" >> ${WORKSPACE}/summary.log
else
    acc=$(grep -Po "Accuracy .* is:\\s+(\\d+(\\.\\d+)?)" ${log_file} | head -n 1 | sed 's/.*://;s/[^0-9.]//g')
    echo "${device};${model};${tasks};${datasets};${acc};${logfile}" >> ${WORKSPACE}/summary.log
fi
