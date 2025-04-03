#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "prepare dassl for xtune"
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git dassl
cd dassl && git fetch origin pull/72/head:xtune && git checkout xtune && cd .. && rsync -avPr dassl/clip/  ./clip_finetune && mv dassl clip_finetune/
echo "dassl done"
