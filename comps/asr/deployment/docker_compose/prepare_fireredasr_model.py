#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import urllib.request
import tarfile
import shutil

def download_and_extract_model(model_url, model_dir, model_name):
    """Download and extract model from URL."""
    print(f"Downloading {model_name} model from {model_url}...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Download the model
    model_path = os.path.join(model_dir, f"{model_name}.tar.gz")
    urllib.request.urlretrieve(model_url, model_path)
    
    # Extract the model
    print(f"Extracting {model_name} model...")
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall(path=model_dir)
    
    # Remove the downloaded tar file
    os.remove(model_path)
    
    print(f"{model_name} model downloaded and extracted successfully.")

def main():
    """Main function to download FireRedASR models."""
    model_dir = os.getenv("FIREREDASR_MODEL_DIR", "/app/pretrained_models")
    model_type = os.getenv("FIREREDASR_MODEL_TYPE", "llm")
    model_version = os.getenv("FIREREDASR_MODEL_VERSION", "latest")
    
    print(f"Preparing FireRedASR models in {model_dir}")
    print(f"Model type: {model_type}")
    print(f"Model version: {model_version}")
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Download models based on type and version
    if model_type == "llm":
        if model_version == "latest":
            # LLM model URLs (replace with actual URLs)
            model_urls = [
                ("https://example.com/models/fireredasr-llm-l/model.pth.tar", model_dir, "model.pth.tar"),
                ("https://example.com/models/fireredasr-llm-l/asr_encoder.pth.tar", model_dir, "asr_encoder.pth.tar"),
                ("https://example.com/models/fireredasr-llm-l/cmvn.ark", model_dir, "cmvn.ark"),
            ]
            
            # Download Qwen2 model (replace with actual URL)
            qwen2_url = "https://example.com/models/qwen2-7b-instruct.tar.gz"
            qwen2_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            download_and_extract_model(qwen2_url, model_dir, "Qwen2-7B-Instruct")
            
        else:
            # Specific version handling
            print(f"Downloading LLM model version {model_version}...")
            # Add version-specific download logic here
            
    elif model_type == "aed":
        if model_version == "latest":
            # AED model URLs (replace with actual URLs)
            model_urls = [
                ("https://example.com/models/fireredasr-aed/model.pth.tar", model_dir, "model.pth.tar"),
                ("https://example.com/models/fireredasr-aed/cmvn.ark", model_dir, "cmvn.ark"),
                ("https://example.com/models/fireredasr-aed/dict.txt", model_dir, "dict.txt"),
                ("https://example.com/models/fireredasr-aed/train_bpe1000.model", model_dir, "train_bpe1000.model"),
            ]
        else:
            # Specific version handling
            print(f"Downloading AED model version {model_version}...")
            # Add version-specific download logic here
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
    
    # Download all models
    for url, target_dir, filename in model_urls:
        model_path = os.path.join(target_dir, filename)
        if not os.path.exists(model_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, model_path)
            print(f"{filename} downloaded successfully.")
        else:
            print(f"{filename} already exists, skipping download.")
    
    print("FireRedASR model preparation completed successfully.")

if __name__ == "__main__":
    main()