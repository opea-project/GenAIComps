# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

MODEL_CONFIG_FILE_MAP = {
    "meta-llama/Llama-2-7b-chat-hf": "./models/llama-2-7b-chat-hf.yaml",
    "mistralai/Mistral-7B-v0.1": "./models/mistral-7b-v0.1.yaml",
}

DATASET_BASE_PATH = "datasets"

CHECK_JOB_STATUS_INTERVAL = 5  # Check every 5 secs

ray_client = None
