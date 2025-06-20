#!/bin/bash
set -e
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
source /GenAIComps/.github/workflows/scripts/change_color
log_dir=/GenAIComps/.github/workflows/scripts/codeScan
ERROR_WARN=false

find . -type f \( -name "Dockerfile*" \) -print -exec hadolint --config /GenAIComps/.github/workflows/scripts/codeScan/.hadolint.yaml {} \; > ${log_dir}/hadolint.log

if [[ $(grep -c "error" ${log_dir}/hadolint.log) != 0 ]]; then
    $BOLD_RED && echo "Error!! Please Click on the artifact button to download and check error details." && $RESET
    echo $(grep "error" ${log_dir}/hadolint.log)
    ERROR_WARN=true
fi

if [[ $(grep -c "warning" ${log_dir}/hadolint.log) != 0 ]]; then
    $BOLD_RED && echo "Warning!! Please Click on the artifact button to download and check warning details." && $RESET
    echo $(grep "warning" ${log_dir}/hadolint.log)
    ERROR_WARN=true
fi

if [ "$ERROR_WARN" = true ]; then
    echo $ERROR_WARN
    exit 1
fi
$BOLD_PURPLE && echo "Congratulations, Hadolint check passed!" && $LIGHT_PURPLE && echo " You can click on the artifact button to see the log details." && $RESET
exit 0
