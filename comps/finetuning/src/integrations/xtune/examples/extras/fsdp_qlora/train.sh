#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# DO NOT use GPTQ/AWQ model in FSDP+QLoRA

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate/fsdp_config.yaml \
    src/train.py examples/extras/fsdp_qlora/llama3_lora_sft.yaml
