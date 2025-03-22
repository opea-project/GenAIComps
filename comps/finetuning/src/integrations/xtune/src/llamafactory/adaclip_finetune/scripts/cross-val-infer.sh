# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [[ "$1" == "msrvtt" ]]; then
   c="io"
else
   c="c"
fi
CUDA_VISIBLE_DEVICES=1 python3 train-s.py --config cfgs/$2-c-32.json --frames_dir data/$2/frames --top_k 16 --frame_agg mlp --do_inference --resume pre-trained/$1-$c-32-16.pth
