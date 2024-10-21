#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

LLM_MODEL_ID="${LLM_MODEL_ID:=Intel/neural-chat-7b-v3-3}"

source /opt/intel/oneapi/setvars.sh
source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --port 9009 \
  --model ${LLM_MODEL_ID} \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --device xpu \
  --enforce-eager \
  $@
