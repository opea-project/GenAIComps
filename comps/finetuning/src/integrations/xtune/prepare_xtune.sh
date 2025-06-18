#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
GUI=$1
if [ -f "done" ]; then
    echo "All component preparation is done"
    echo "Please follow README.md to install driver and other dependency"
    if [ -z $GUI ]; then
        ps aux | grep "llamafactory-cli webui" | head -n 1 | awk '{print $2}' |xargs kill
        ZE_AFFINITY_MASK=0 llamafactory-cli webui &
        if [ $? -eq 0 ]; then
            echo "server start successfully"
        else
            echo "failed to start server, please check your environment"
        fi
    else
        ps aux | grep "llamafactory-cli webui" | head -n 1 | awk '{print $2}' |xargs kill
        echo "kill webui service done"
    fi
else
    echo "start prepare for xtune"
    bash clip_finetune/prepare_clip_finetune.sh
    bash adaclip_finetune/prepare_adaclip_finetune.sh
    mv llama_factory/src .
    bash llama_factory/prepare_llama_factory.sh
    rm -rf llama_factory
    rsync -avPr clip_finetune src/llamafactory/
    rsync -avPr adaclip_finetune src/llamafactory/
    rm -rf clip_finetune adaclip_finetune
    echo "prepare for xtune done"
    echo "install requirements"
    python -m pip install --no-cache-dir -r requirements.txt
    pip uninstall -y torchvision torchaudio
    pip install setuptools==75.6.0
    python -m pip install --no-cache-dir torch==2.6.0+xpu torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    cd src/llamafactory/clip_finetune/dassl
    python setup.py develop
    cd ../../../..
    pip install matplotlib
    pip install -e ".[metrics]"
    pip install --no-cache-dir --force-reinstall intel-extension-for-pytorch==2.6.10+xpu oneccl_bind_pt==2.6.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    pip install "transformers>=4.50.0" optimum "auto_gptq>=0.5.0"
    echo "start llamafactory webui"
    if [ -z $GUI ]; then
        ZE_AFFINITY_MASK=0 llamafactory-cli webui &
        if [ $? -eq 0 ]; then
            echo "server start successfully"
        else
            echo "failed to start server, please check your environment"
        fi
    else
        ps aux | grep "llamafactory-cli webui" | head -n 1 | awk '{print $2}' |xargs kill
        echo "kill webui service done"
    fi
    echo 0 >> "done"
    echo "Please follow README.md to install driver or update torch lib"
fi
