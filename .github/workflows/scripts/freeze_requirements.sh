#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function freeze() {
    local file=$1
    local folder=$(dirname "$file")
    pip-compile \
        --no-upgrade \
        --output-file \
        --no-annotate \
        --no-header \
        "$folder/freeze.txt" "$file"
    if [[ -e "$folder/freeze.txt" ]]; then
        if [[ "$keep_origin_packages" == "true" ]]; then
            filter_packages
        else
            mv "$folder/freeze.txt" "$file"
        fi
        exit 1
    fi
}


function filter_packages() {
    packages1=$(cut -d'=' -f1 "$file" | sort)
    packages2=$(cut -d'=' -f1 "$folder/freeze.txt" | sort)
    common_packages=$(comm -12 <(echo "$packages1") <(echo "$packages2"))
    rm "$file"
    while IFS= read -r line; do
        package=$(echo "$line" | cut -d'=' -f1)
        if echo "$common_packages" | grep -q "^$package$"; then
            echo "$line" >>"$file"
        fi
    done <"$folder/freeze.txt"
    rm "$folder/freeze.txt"
}


function check_branch_name() {
    branch_name=$(git branch --show-current)
    if [[ "$branch_name" == *"rc" ]]; then
        echo "$branch_name is release branch"
    else
        exit 0
    fi
}


function main() {
    check_branch_name
    pip install pip-tools --upgrade
    export -f freeze
    find . -name "requirements.txt" | xargs -n 1 -I {} bash -c 'freeze "$@"' _ {}
}

main
