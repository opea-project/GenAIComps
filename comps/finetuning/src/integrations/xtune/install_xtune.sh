#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [ -f "README.md" ]; then
    echo "All component preparation is done"
    echo "Please follow README.md to install driver and other dependency"
else
    echo "prepare dassl for xtune"
    git clone https://github.com/KaiyangZhou/Dassl.pytorch.git dassl
    cd dassl && git apply --reject ../dassl-update-for-xtune.patch && cd ..
    mv dassl clip_finetune/
    echo "dassl done"
    echo "prepare adaclip for xtune"
    git clone https://github.com/SamsungLabs/AdaCLIP.git
    cd AdaCLIP && git apply --reject ../adaclip-update-for-xtune.patch && cd .. && rsync -avPr AdaCLIP/  adaclip_finetune/ && rm -rf AdaCLIP
    echo "adaclip done"
    echo "prepare llama-factory for xtune"
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory && git checkout 38e955d4a917a7e600cfd17a041a1472e8d81370 && git apply --reject ../update_for_xtune.patch && cd ..
    rsync -avPr LLaMA-Factory/  .
    rm -rf LLaMA-Factory
    mv clip_finetune src/llamafactory/
    mv adaclip_finetune src/llamafactory/
    echo "prepare for xtune done"
    echo "Please follow README.md to install driver and other dependency"
fi
