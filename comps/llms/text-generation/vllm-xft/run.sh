#!/bin/sh

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# check xfastertransformer installation
export $(python -c 'import xfastertransformer as xft; print(xft.get_env())')

# download model
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct

export HF_DATASET_DIR=./Qwen2-7B-Instruct
export OUTPUT_DIR=./output
export TOKEN_PATH=./Qwen2-7B-Instruct

# convert the model to fastertransformer format
python -c 'import os; import xfastertransformer as xft; xft.Qwen2Convert().convert(os.environ["HF_DATASET_DIR"], os.environ["OUTPUT_DIR"])'

# serving with vllm
python -m vllm.entrypoints.openai.api_server \
        --model ${OUTPUT_DIR} \
        --tokenizer ${TOKEN_PATH} \
        --dtype bf16 \
        --kv-cache-dtype fp16 \
        --served-model-name xft \
        --port 18688 \
        --trust-remote-code &

# run llm microservice wrapper
python -m llm.py
