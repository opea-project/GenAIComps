#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if [ -f "done" ]; then
    echo "All component preparation is done"
    echo "Please follow README.md to install driver and other dependency"
else
    echo "start prepare for xtune"
    bash clip_finetune/prepare_clip_finetune.sh
    bash adaclip_finetune/prepare_adaclip_finetune.sh
    cd llama_factory && mv examples src ../ && cd ..
    bash llama_factory/prepare_llama_factory.sh
    rm -rf llama_factory
    rsync -avPr clip_finetune src/llamafactory/
    rsync -avPr adaclip_finetune src/llamafactory/
    rm -rf clip_finetune adaclip_finetune
    echo "prepare for xtune done"
    echo 0 >> done
    echo "Please follow README.md to install driver and other dependency"
fi
