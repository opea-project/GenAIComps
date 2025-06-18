#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Download default voice reference
wget "https://github.com/intel/intel-extension-for-transformers/raw/refs/heads/main/intel_extension_for_transformers/neural_chat/assets/audio/welcome_cn.wav"

# Download models
MODEL_REPO=https://huggingface.co/lj1995/GPT-SoVITS
llm_download=${llm_download}
echo "llm_download: ${llm_download}"
if [ "$llm_download" = "True" ]; then
  # clean if exists
  rm -rf /home/user/GPT-SoVITS/GPT_SoVITS/pretrained_models/*

  echo "Please wait for model download..."
  git lfs install &&  git clone --depth 1 --branch main --single-branch ${MODEL_REPO} /home/user/pretrained_models
  rm -rf /home/user/pretrained_models/.git
  mv /home/user/pretrained_models/*  /home/user/GPT-SoVITS/GPT_SoVITS/pretrained_models/
  rm -rf /home/user/pretrained_models
elif [ "$llm_download" = "False" ]; then
  echo "No model download"
else
  echo "llm_download should be True or False"
  exit 1
fi

python api.py --default_refer_path ./welcome_cn.wav --default_refer_text "欢迎使用" --default_refer_language "zh"
