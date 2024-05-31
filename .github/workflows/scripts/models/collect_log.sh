#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail
set -x
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

log_file="/log/${device}/${model}/${device}-${tasks}-${model}-${datasets}.log"
$BOLD_YELLOW && echo "-------- Collect logs --------" && $RESET

echo "working in"
pwd
if [[ ! -f ${log_file} ]]; then
    echo "${device};${model};${tasks};${datasets};;" >> ${WORKSPACE}/summary.log
else
    acc=$(grep -Po "acc .*(\d+(\.\d+)?)" ${log_file} | awk -F "|" '{print $2}' | head -n 1 | sed 's/.*://;s/[^0-9.]//g')
    echo "${device};${model};${tasks};${datasets};${acc};" >> ${WORKSPACE}/summary.log
fi
