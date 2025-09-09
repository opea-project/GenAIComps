#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

# Use pip-tools to compile all requirements.in files to requirements.txt files
# req_files=($(find . -type f -name "requirements.in"))

# Manually specify the requirements.in files to avoid some unnecessary ones
req_files=(
# ./comps/third_parties/video-llama/src/requirements.in
# ./comps/agent/src/requirements.in
./requirements.in
)

# For loop each requirements.in file and compile it to requirements.txt, requirements-gpu.txt, and requirements-cpu.txt if the corresponding files exist
for each_file in "${req_files[@]}"; do
  if [ -f "${each_file%.in}.txt" ]; then
    uv pip compile "$each_file" --universal -o "${each_file%.in}.txt" --upgrade
  fi
  if [ -f "${each_file%.in}-gpu.txt" ]; then
    uv pip compile "$each_file" --universal -o "${each_file%.in}-gpu.txt" --upgrade
  fi
  if [ -f "${each_file%.in}-cpu.txt" ]; then
    uv pip compile --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "$each_file" --universal -o "${each_file%.in}-cpu.txt" --upgrade
    # CPU version of torch will be install separately in the dockerfile without nv related dependency to reduce the image size.
    # If torch is in the requirements-cpu.txt file, remove the lines that contain '# via torch' and the lines that are dependencies of torch.
    if [[ $(grep 'torch==' ${each_file%.in}-cpu.txt) ]]; then
        req_path=${each_file%.in}-cpu.txt
        awk 'prev && /# via torch/ { prev=""; next } { if (prev) print prev; prev=$0 } END { if (prev) print prev }' ${req_path} | grep -v '# via torch' > tmp && mv tmp ${req_path}
        awk 'skip == 1 && $0 ~ /^[[:space:]]*#/ { next } skip == 1 && $0 !~ /^[[:space:]]*#/ { skip = 0 } $0 ~ /^torch/ { skip = 1; next } skip == 0 { print }' ${req_path} > tmp && mv tmp ${req_path}
        sed -i '/ torch/d' ${req_path}
    fi
  fi
done
