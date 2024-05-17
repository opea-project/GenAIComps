#!/bin/bash

# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set default values
default_port=8000
default_model="meta-llama/Llama-2-7b-hf"
swap_space=50

# Assign arguments to variables
port_number=${1:-$default_port}
model_name=${2:-$default_model}


# Check if all required arguments are provided
if [ "$#" -lt 0 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 [port_number] [model_name]"
    exit 1
fi

# Set the Huggingface cache directory variable
HF_CACHE_DIR=$HOME/.cache/huggingface

# Start the model server with openvino as the backened inference server
docker run --rm --name="vllm-openvino-server" -p $port_number:$port_number -v $HF_CACHE_DIR:/root/.cache/huggingface vllm:openvino --model $model_name --port $port_number --disable-log-requests --swap-space $swap_space
