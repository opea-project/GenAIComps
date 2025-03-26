#!/usr/bin/env bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#!/bin/bash

#LLM_MODEL_ID mush be a model path
llm_name=$LLM_MODEL_ID
WORKPATH="/home/user/comps/llms/src/text-generation/"

if [[ $llm_name == *"Phi-4-multimodal-instruct"* ]]; then
    cd $WORKPATH
    echo -e "Patching into the multimodal models"
    cp patch/phi4-multimodal-patch/*.py $llm_name/
    export PT_HPU_LAZY_MODE=1
elif [[ $llm_name == *"Phi-4-mini-instruct"* ]]; then
    cd $WORKPATH
    git clone -b transformers_future https://github.com/huggingface/optimum-habana
    cd optimum-habana
    cp ../patch/optimum-habana-phi4.patch .
    git apply optimum-habana-phi4.patch
    pip install -e .
    cd examples/text-generation/
    pip install -r requirements.txt
    cd phi-4-mini-instruct/
    bash ./01-patch-transformer.sh
fi

cd $WORKPATH
python opea_llm_microservice.py
