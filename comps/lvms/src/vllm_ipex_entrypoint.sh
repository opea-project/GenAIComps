#!/bin/sh

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [ "$LOAD_QUANTIZATION" = "None" ]; then
    echo "LOAD_QUANTIZATION is None, will load the model without online quantization."

    TORCH_LLM_ALLREDUCE=1 \
    VLLM_USE_V1=1 \
    CCL_ZE_IPC_EXCHANGE=pidfd \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python3 -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL_NAME} \
        --dtype=float16 \
        --enforce-eager \
        --port 8000 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --gpu-memory-util=0.9 \
        --max-num-batched-tokens=${MAX_MODEL_LEN} \
        --disable-log-requests \
        --max-model-len=${MAX_MODEL_LEN} \
        --block-size 64 \
        -tp=${TENSOR_PARALLEL_SIZE} \
        --enable_prefix_caching
else
    echo "LOAD_QUANTIZATION is $LOAD_QUANTIZATION"

    TORCH_LLM_ALLREDUCE=1 \
    VLLM_USE_V1=1 \
    CCL_ZE_IPC_EXCHANGE=pidfd \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python3 -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --served-model-name ${SERVED_MODEL_NAME} \
        --dtype=float16 \
        --enforce-eager \
        --port 8000 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --gpu-memory-util=0.9 \
        --max-num-batched-tokens=${MAX_MODEL_LEN} \
        --disable-log-requests \
        --max-model-len=${MAX_MODEL_LEN} \
        --block-size 64 \
        --quantization ${LOAD_QUANTIZATION} \
        -tp=${TENSOR_PARALLEL_SIZE} \
        --enable_prefix_caching
fi
