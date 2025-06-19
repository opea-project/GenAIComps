#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

function prepare_models() {

    if [ $# -lt 2 ]; then
        echo "Usage: prepare_models <directory> <model1> [model2] ..."
        return
    fi

    local model_path=$1
    shift
    mkdir -p ${model_path}
    python3 -m pip install huggingface_hub[cli] --user
    # Workaround for huggingface-cli reporting error when set --cache-dir to same as default
    local extra_args=""
    local default_model_dir=$(readlink -m ~/.cache/huggingface/hub)
    local real_model_dir=$(echo ${model_path/#\~/$HOME} | xargs readlink -m )
    if [[ "${default_model_dir}" != "${real_model_dir}" ]]; then
        extra_args="--cache-dir ${model_path}"
    fi
    for m in "$@"; do
      PATH=~/.local/bin:$PATH huggingface-cli download ${extra_args} $m
    done
}
