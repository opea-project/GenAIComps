#!/bin/bash

# Copyright (c) 2024 Advanced Micro Devices, Inc.

docker build -f Dockerfile.amd_gpu -t opea/llm-vllm-rocm:latest . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
