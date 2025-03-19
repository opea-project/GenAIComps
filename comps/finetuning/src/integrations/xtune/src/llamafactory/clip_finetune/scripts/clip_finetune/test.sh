#!/bin/bash

# custom config
bash scripts/CLIP_finetune/clip_adapter_hf.sh flickr5k vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_bias_hf.sh flickr5k vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_fullfinetune_hf.sh flickr5k vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_prompt_hf.sh flickr5k vit_b16 1 True 0 cuda 1

bash scripts/CLIP_finetune/clip_adapter_hf.sh mscoco vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_bias_hf.sh mscoco vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_fullfinetune_hf.sh mscoco vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_prompt_hf.sh mscoco vit_b16 1 True 0 cuda 1

bash scripts/CLIP_finetune/clip_adapter_hf.sh mini_imagenet vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_bias_hf.sh mini_imagenet vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_fullfinetune_hf.sh mini_imagenet vit_b16 0 cuda 1
bash scripts/CLIP_finetune/clip_prompt_hf.sh mini_imagenet vit_b16 1 True 0 cuda 1