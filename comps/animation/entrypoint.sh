#!/bin/sh
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Environment variables
export PT_HPU_LAZY_MODE=0
export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1

# Wav2Lip, GFPGAN
python3 animation.py \
--port $((ANIMATION_PORT)) \
--inference_mode $INFERENCE_MODE \
--checkpoint_path $CHECKPOINT_PATH \
--face $FACE \
--audio $AUDIO \
--outfile $OUTFILE \
--img_size $((FACESIZE)) \
-v $GFPGAN_MODEL_VERSION \
-s $((UPSCALE_FACTOR)) \
--fps $((FPS)) \
--only_center_face \
--bg_upsampler None
