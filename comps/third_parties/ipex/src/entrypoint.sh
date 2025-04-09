#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Collect all command-line arguments into an array
CMDS=()
while [ $# -gt 0 ]; do
  CMDS+=("$1")
  shift
done

# Activate the Miniforge environment
. ~/miniforge3/bin/activate
conda activate py310

# Set environment variables for oneCCL bindings for PyTorch
TMP=$(python -c "import torch; import os; print(os.path.abspath(os.path.dirname(torch.__file__)))")
. ${TMP}/../oneccl_bindings_for_pytorch/env/setvars.sh

# Print a performance note
echo "**Note:** For better performance, please consider to launch workloads with command 'ipexrun'."

# Run the inference script
python /root/ipex_inference.py "${CMDS[@]}"
