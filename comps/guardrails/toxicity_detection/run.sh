#!/bin/sh

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Preloading libiomp5.so is essential for optimal performance.
# libiomp5.so is the Intel OpenMP runtime library, providing parallel computation support,
# thread management, task scheduling, and performance optimization on Intel X86 platforms.

# run toxicity detection microservice 
python toxicity_detection.py
