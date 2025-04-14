#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "prepare llama-factory for xtune"
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && git fetch origin pull/7625/head:xtune && git checkout xtune && cd ..
rsync -avPr LLaMA-Factory/  .
rm -rf LLaMA-Factory
echo "prepare llama-factory done"
