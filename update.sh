#!/bin/bash

req_files=(
./comps/retrievers/src/requirements.in
./comps/third_parties/pathway/src/requirements.in
)

for each_file in "${req_files[@]}"; do
  if [ -f "${each_file%.in}.txt" ]; then
    uv pip compile --python=$(which python3.11)  "$each_file" --universal -o "${each_file%.in}.txt" --upgrade
  fi
  if [ -f "${each_file%.in}-gpu.txt" ]; then
    uv pip compile --python=$(which python3.11)  "$each_file" --universal -o "${each_file%.in}-gpu.txt" --upgrade
  fi
  if [ -f "${each_file%.in}-cpu.txt" ]; then
    uv pip compile --python=$(which python3.11)  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "$each_file" --universal -o "${each_file%.in}-cpu.txt" --upgrade
    if [[ $(grep 'torch==' ${each_file%.in}-cpu.txt) ]]; then
        req_path=${each_file%.in}-cpu.txt
        awk 'prev && /# via torch/ { prev=""; next } { if (prev) print prev; prev=$0 } END { if (prev) print prev }' ${req_path} | grep -v '# via torch' > tmp && mv tmp ${req_path}
        awk 'skip == 1 && $0 ~ /^[[:space:]]*#/ { next } skip == 1 && $0 !~ /^[[:space:]]*#/ { skip = 0 } $0 ~ /^torch/ { skip = 1; next } skip == 0 { print }' ${req_path} > tmp && mv tmp ${req_path}
        sed -i '/ torch/d' ${req_path}
    fi
  fi
done