#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

pip install pip-tools --upgrade

function freeze() {
    local file=$1
    local folder=$(dirname "$file")
    pip-compile --no-upgrade --output-file "$folder/freeze.txt" "$file"
    if [ -e "$folder/freeze.txt" ]; then
        mv "$folder/freeze.txt" "$file"
    fi
}

export -f freeze

find . -name "requirements.txt"
find . -name "requirements.txt" | xargs -n 1 -I {} bash -c 'freeze "$@"' _ {}
