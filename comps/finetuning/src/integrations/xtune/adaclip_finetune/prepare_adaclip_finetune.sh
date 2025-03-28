#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo "prepare adaclip for xtune"
git clone https://github.com/SamsungLabs/AdaCLIP.git
cd AdaCLIP && git fetch origin pull/3/head:xtune && git checkout xtune && cd .. && rsync -avPr AdaCLIP/  adaclip_finetune/ && rm -rf AdaCLIP
echo "adaclip done"
