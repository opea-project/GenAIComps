# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/msrvtt-jsfusion.json --frames_dir data/msrvtt/frames --top_k 12 --frame_agg mlp --do_inference --resume updates/trained_model_20241022.pth
CUDA_VISIBLE_DEVICES=1 python3 train-s.py --config $1 --frames_dir $2 --top_k 16 --frame_agg mlp --do_inference --resume $3
