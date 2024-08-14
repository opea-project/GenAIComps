#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function freeze() {
    local file=$1
    local folder=$(dirname "$file")
    local keep_origin_packages="true"
    pip-compile \
        --no-upgrade \
        --no-annotate \
        --no-header \
        --output-file "$folder/freeze.txt" \
        "$file"
    if [[ -e "$folder/freeze.txt" ]]; then
        if [[ "$keep_origin_packages" == "true" ]]; then
            packages1=$(cut -d'=' -f1 "$file" | sort)
            packages2=$(cut -d'=' -f1 "$folder/freeze.txt" | sort)
            common_packages=$(comm -12 <(echo "$packages2" | tr '[:upper:]' '[:lower:]' | sed 's/[-_]/-/g') <(echo "$packages1" | tr '[:upper:]' '[:lower:]' | sed 's/[-_]/-/g'))

            rm "$file"
            while IFS= read -r line; do
                package=$(echo "$line" | cut -d'=' -f1)
                package_transformed=$(echo "$package" | tr '[:upper:]' '[:lower:]' | sed 's/[_-]/-/g')
                pattern=$(echo "$package_transformed" | sed 's/\[/\\\[/g; s/\]/\\\]/g')
                if echo "$common_packages" | grep -q "^$pattern$"; then
                    echo "$line" >>"$file"
                fi
            done <"$folder/freeze.txt"
            rm "$folder/freeze.txt"
        else
            mv "$folder/freeze.txt" "$file"
        fi
        exit 1
    fi
}

function check_branch_name() {
    branch_name=$(git branch --show-current)
    if [[ "$branch_name" == *"rc" ]]; then
        echo "$branch_name is release branch"
    else
        echo "$branch_name is not release branch" && exit 0
    fi
}

function main() {
    check_branch_name
    pip install pip-tools --upgrade
    export -f freeze
    find . -name "requirements.txt" | xargs -n 1 -I {} bash -c 'freeze "$@"' _ {}
}

main
