# XTune - Model finetune tool for Intel GPU

**`Xtune`** is an model finetune tool for Intel GPU(Intel Arc 770)

> [!NOTE]
>
> - _`Xtune`_ incorporates with Llama-Factory to offer various methods for finetuning visual models (CLIP, AdaCLIP), LLM and Multi-modal models​. It makes easier to choose the method and to set fine-tuning parameters.

The core features include:

- Four finetune method for CLIP, details in [CLIP](./doc/key_features_for_clip_finetune_tool.md)
- Three finetune method for AdaCLIP, details in [AdaCLIP](./doc/adaclip_readme.md)
- Automatic hyperparameter searching enabled by Optuna [Optuna](https://github.com/optuna/optuna)
- Distillation from large models with Intel ARC GPU​
- Incorporate with Llama-Factory UI​
- Finetune methods for multi-modal models (to be supported)​

You can use this UI to easily access basic functions(merge two tool into one UI),

or use the command line to use tools separately which is easier to customize parameters and has more comprehensive functionality.

## Installation

> [!IMPORTANT]
> Installation is mandatory.

> Please install git first and make sure `git clone` can work.

> Please fololow [install_dependency](./doc/install_dependency.md) to install Driver for Arc 770

### 1. Install xtune on native

Run install_xtune.sh to prepare component.

```bash
conda create -n xtune python=3.10 -y
conda activate xtune
apt install -y rsync
# open webui as default
bash prepare_xtune.sh
# this way it will not open webui
# bash prepare_xtune.sh false
```

Blow command is in prepare_xtune.sh. You can ignore it if you don't want to update lib manually.

```bash
# if you want to run on NVIDIA GPU
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# else run on A770
# You can refer to https://github.com/intel/intel-extension-for-pytorch for latest command to update lib
    python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    python -m pip install intel-extension-for-pytorch==2.5.10+xpu oneccl_bind_pt==2.5.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Install xtune on docker

#### 2.1 Build Docker Image

Build docker image with below command:

```bash
cd ../../../deployment/docker_compose
export DATA="where to find dataset"
docker build -t opea/finetuning-xtune:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy --build-arg HF_TOKEN=$HF_TOKEN --build-arg DATA=$DATA -f comps/finetuning/src/Dockerfile.xtune .
```

#### 2.2 Run Docker with CLI

Suse docker compose with below command:

```bash
export HF_TOKEN=${your_huggingface_token}
export DATA="where to find dataset"
cd ../../../deployment/docker_compose
docker compose -f compose.yaml up finetuning-xtune -d
```

## Data Preparation

Please refer to [data/Prepare_dataset.md](./doc/Prepare_dataset.md) for checking the details about the dataset files.

> [!NOTE]
> Please update `dataset_info.json` to use your custom dataset.

Prepare dataset info for caltech101
make `caltech101.json` in your dataset directory

```json
[]
```

then make `dataset_info.json` in your dataset directory

```json
{
  "caltech101": {
    "file_name": "caltech101.json"
  }
}
```

## Fine-Tuning with LLaMA Board GUI (powered by [Gradio](https://github.com/gradio-app/gradio))

> [!NOTE] We don't support multi-card in GUI now, will add it later.

When run with prepare_xtune.sh, it will automatic run ZE_AFFINITY_MASK=0 llamafactory-cli webui.

If you see "server start successfully" in terminal.
You can access in web through http://localhost:7860/

The UI component information can be seen in doc/ui_component.md after run with prepare_xtune.sh.

When run with prepare_xtune.sh, it will automatic run ZE_AFFINITY_MASK=0 llamafactory-cli webui.

If you see "server start successfully" in terminal.
You can access in web through http://localhost:7860/

The UI component information can be seen in doc/ui_component.md after run with prepare_xtune.sh.

```bash
 Run with A100:
 CUDA_VISIBLE_DEVICES=0 llamafactory-cli webui
 Run with ARC770:
 ZE_AFFINITY_MASK=0 llamafactory-cli webui

 Then access in web through http://localhost:7860/
```

## Fine-Tuning with Shell instead of GUI

After run `prepare_xtune.sh`, it will download all related file. And open webui as default.

You can run `bash prepare_xtune.sh false` to close webui. Then you can run fine-tune with shell.

Below are examples.

### CLIP

Please see [doc](./doc/key_features_for_clip_finetune_tool.md) for how to config feature

```bash
cd src/llamafactory/clip_finetune
# Please see README.md in src/llamafactory/clip_finetune for detail
```

### AdaCLIP

```bash
cd src/llamafactory/adaclip_finetune
# Please see README.md in src/llamafactory/adaclip_finetune for detail
```

### DeepSeek-R1 Distillation(not main function)

Please see [doc](./doc/DeepSeek-R1_distillation_best_practice-v1.3.pdf) for details

#### Step 1: Download existing CoT synthetic dataset from huggingface

Dataset link: https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B

#### Step 2: Convert to sharegpt format

```bash
cd data
import json
from datasets import load_dataset
# Load the dataset
dataset = load_dataset("Magpie-Align/Magpie-Reasoning-V2-""250K-CoT-Deepseek-R1-Llama-70B")
dataset = dataset["train"]
# Filter dataset
## Change the filter conditions according to your needs
dataset = dataset.filter(lambda example: len(example['response']) <= 1024)
# Save as sharegpt format
with open("Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B-response1024.json",
'w') as f:
  json.dump(list(dataset), f, ensure_ascii=False, indent=4)
```

#### Step 3: Register CoT dataset LLAMA-Factory dataset_info.json

```bash
cd data
vim dataset_info.json

# make sure the file is put under `xtune/data`
"deepseek-r1-distill-sample": {
  "file_name": "Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B-response1024.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  }
}
```

#### Step 4: Use the accelerate command to enable training on XPU plugin

```
accelerate config

For Single GPU:
  Which type of machine are you using?
  No distributed training
  Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
  Do you want to use XPU plugin to speed up training on XPU? [yes/NO]:yes
  Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
  Do you want to use DeepSpeed? [yes/NO]: NO
  What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:all
  Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]:
  Do you wish to use mixed precision?
  bf16
For Multi-GPU with FSDP:
  Which type of machine are you using?
  multi-XPU
  How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
  Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: NO
  Do you want to use XPU plugin to speed up training on XPU? [yes/NO]:yes
  Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
  Do you want to use DeepSpeed? [yes/NO]: NO
  Do you want to use FullyShardedDataParallel? [yes/NO]: yes
  What should be your sharding strategy?
  FULL_SHARD
  Do you want to offload parameters and gradients to CPU? [yes/NO]: NO
  What should be your auto wrap policy?
  TRANSFORMER_BASED_WRAP
  Do you want to use the model's `_no_split_modules` to wrap. Only applicable for Transformers [yes/NO]: yes
  What should be your FSDP's backward prefetch policy?
  BACKWARD_PRE
  What should be your FSDP's state dict type?
  SHARDED_STATE_DICT
  Do you want to enable FSDP's forward prefetch policy? [yes/NO]: yes
  Do you want to enable FSDP's `use_orig_params` feature? [YES/no]: yes
  Do you want to enable CPU RAM efficient model loading? Only applicable for Transformers models. [YES/no]: yes
  Do you want to enable FSDP activation checkpointing? [yes/NO]: yes
  How many GPU(s) should be used for distributed training? [1]:2
  Do you wish to use mixed precision?
  bf16
```

#### Step 5: Run with train script as follows

```bash
export ONEAPI_DEVICE_SELECTOR="level_zero:0"
MODEL_ID="microsoft/Phi-3-mini-4k-instruct"
EXP_NAME="Phi-3-mini-4k-instruct-r1-distill-finetuned"
DATASET_NAME="deepseek-r1-distill-sample"
export OUTPUT_DIR="where to put output"
accelerate launch src/train.py --stage sft --do_train --use_fast_tokenizer --new_special_tokens "<think>,</think>" --resize_vocab --flash_attn auto --model_name_or_path ${MODEL_ID} --dataset ${DATASET_NAME} --template phi --finetuning_type lora --lora_rank 8 --lora_alpha 16 --lora_target q_proj,v_proj,k_proj,o_proj --additional_target lm_head,embed_tokens --output_dir $OUTPUT_DIR --overwrite_cache --overwrite_output_dir --warmup_steps 100 --weight_decay 0.1 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --ddp_timeout 9000 --learning_rate 5e-6 --lr_scheduler_type cosine --logging_steps 1 --save_steps 1000 --plot_loss --num_train_epochs 3 --torch_empty_cache_steps 10 --bf16
```

## `Xtune` Examples

See screenshot of running CLIP and AdaCLIP finetune on Intel Arc A770 in README_XTUNE.md.

## Citation

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```

## Acknowledgement

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [CLIP-Adapter](https://github.com/gaopengcuhk/CLIP-Adapter) and [CoOp](https://github.com/KaiyangZhou/Dassl.pytorch). Thanks for their wonderful works.
