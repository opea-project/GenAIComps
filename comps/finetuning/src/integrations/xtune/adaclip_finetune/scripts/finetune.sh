# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [[ "$1" == "msrvtt" ]]; then
   c1="io"
else
   c1="c"
fi
if [[ "$2" == "msrvtt" ]]; then
   c2="io"
else
   c2="c"
fi
CUDA_VISIBLE_DEVICES=1 python3 train-s.py --config cfgs/finetune/$2-finetune-$3-$c2-32.json --frames_dir data/$2/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume pre-trained/$1-$c1-32-16.pth --num_epochs $4 # --learning_rate 1e-7 --coef_lr 0.1
