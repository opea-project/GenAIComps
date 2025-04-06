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
bash prepare_xtune.sh
```

Blow command is in prepare_xtune.sh. You can ignore it if you don't want to update lib manually.

```bash
pip install -r requirements.txt
# if you want to run on NVIDIA GPU
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# else run on A770
# You can refer to https://github.com/intel/intel-extension-for-pytorch for latest command
    python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

cd src/llamafactory/clip_finetune/dassl
python setup.py develop
cd ../../../..
pip install matplotlib
pip install -e ".[metrics]"
pip install --no-deps transformers==4.45.0 datasets==2.21.0 accelerate==0.34.2 peft==0.12.0
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

```bash
 Run with A100:
 CUDA_VISIBLE_DEVICES=0 llamafactory-cli webui
 Run with ARC770:
 ZE_AFFINITY_MASK=0 llamafactory-cli webui

 Then access in web through http://localhost:7860/
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
