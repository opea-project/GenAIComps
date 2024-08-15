#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function freeze() {
    local file=$1
    local folder=$(dirname "$file")
    local keep_origin_packages="true"
    echo "::group::Check $file ..."
    pip-compile \
        --no-upgrade \
        --no-annotate \
        --no-header \
        --output-file "$folder/freeze.txt" \
        "$file"
    echo "::endgroup::"
    if [[ -e "$folder/freeze.txt" ]]; then
        if [[ "$keep_origin_packages" == "true" ]]; then
            sed -i '/^\s*#/d; s/#.*//; /^\s*$/d' "$file"
            sed -i '/^\s*#/d; s/#.*//; /^\s*$/d' "$folder/freeze.txt"

            packages1=$(cut -d'=' -f1 "$file" | tr '[:upper:]' '[:lower:]' | sed 's/[-_]/-/g')
            packages2=$(cut -d'=' -f1 "$folder/freeze.txt" | tr '[:upper:]' '[:lower:]' | sed 's/[-_]/-/g')
            common_packages=$(comm -12 <(echo "$packages2" | sort) <(echo "$packages1" | sort))

            rm "$file"
            while IFS= read -r line; do
                package=$(echo "$line" | cut -d'=' -f1)
                package_transformed=$(echo "$package" | tr '[:upper:]' '[:lower:]' | sed 's/[_-]/-/g')
                pattern=$(echo "$package_transformed" | sed 's/\[/\\\[/g; s/\]/\\\]/g')
                if echo "$common_packages" | grep -q "^$pattern$"; then
                    echo "$line" >>"$file"
                fi
            done <"$folder/freeze.txt"
            # rm "$folder/freeze.txt"
        else
            mv "$folder/freeze.txt" "$file"
        fi
        exit 1
    fi
}

function check_branch_name() {
    if [[ "$GITHUB_REF_NAME" == *"rc" ]]; then
        echo "$GITHUB_REF_NAME is release branch"
    else
        echo "$GITHUB_REF_NAME is not release branch"
        # exit 0
    fi
}

function main() {
    check_branch_name
    pip install pip-tools --upgrade
    export -f freeze
    find . -name "requirements.txt" | xargs -n 1 -I {} bash -c 'freeze "$@"' _ {}
}

main
