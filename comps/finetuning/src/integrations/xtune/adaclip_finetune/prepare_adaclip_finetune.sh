#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "prepare adaclip for xtune"
git clone https://github.com/caoyunkang/AdaCLIP.git
cd AdaCLIP && git fetch origin pull/45/head:xtune && git checkout xtune && cd .. && rsync -avPr AdaCLIP/  adaclip_finetune/ && rm -rf AdaCLIP
echo "adaclip done"
