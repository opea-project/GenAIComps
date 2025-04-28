#!/bin/sh

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

. ~/miniforge3/bin/activate && conda activate sglang
export LD_PRELOAD=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libiomp5.so:${CONDA_PREFIX:-"$(dirname $(which conda))/../"}/lib/libtcmalloc.so
SGLANG_CPU_OMP_THREADS_BIND=${SGLANG_CPU_OMP_THREADS_BIND} python3 -m sglang.launch_server --model ${MODEL_ID} --trust-remote-code --device cpu --disable-overlap-schedule  --tp ${TP_SIZE} --chunked-prefill-size 2048 --max-running-requests 32 --mem-fraction-static 0.8 --context-length 65536 --max-total-tokens 65536  --port ${SGLANG_LLM_PORT} --chat-template llama-4