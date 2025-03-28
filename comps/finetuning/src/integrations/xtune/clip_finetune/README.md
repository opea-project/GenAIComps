## Requirements

We utilize the code base of [CoOp](https://github.com/KaiyangZhou/CoOp). Please follow their instructions to prepare the environment and datasets.

## run on NV A100

```python
conda create -y -n clip_adapter python=3.10
conda activate clip_adapter
cd Dassl

#This step can be replaced by [ubstakk](#section1) if you run on Intel Arc 770
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install dassl library (no need to re-build if the source code is modified)
python setup.py develop

cd ..

# Install dependencies for clip
pip install -r requirements.txt
pip install transformers
export HF_ENDPOINT=https://hf-mirror.com
```

## run on A770

### Download oneapi

You can refer to https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

```python
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/dfc4a434-838c-4450-a6fe-2fa903b75aa7/intel-oneapi-base-toolkit-2025.0.1.46_offline.sh
sudo sh ./intel-oneapi-base-toolkit-2025.0.1.46_offline.sh -a --silent --cli --eula accept
```

### Install Driver

please follow [Install Dependency](./doc/install_dependency.md) to install public Driver

### Install IPEX and other lib

```python
conda create -y -n clip_adapter python=3.10
conda activate clip_adapter
cd Dassl
# You can refer to https://github.com/intel/intel-extension-for-pytorch for latest command
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu oneccl_bind_pt==2.5.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

```python
#Install dependencies
pip install -r requirements.txt

# Install dassl library (no need to re-build if the source code is modified)
python setup.py develop

cd ..

# Install dependencies for clip
pip install -r requirements.txt
pip install transformers==4.41.2
export HF_ENDPOINT=https://hf-mirror.com
```

# Prepare Dataset

Please follow [doc](./doc/Prepare_dataset.md)

```python
# support  caltech101, mini-imagenet, flickr30k, flickr5k
The dataset directory should be link
data
--- caltech-101
------ 101_Objectcatehories
------ split_zhou_Caltech101.json
--- flickr5k
--- mini-imagenet
```

## Get Started

Run on Nvidia

```python
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export DATA=/path/to/dataset
# run clip_adapter
# run with huggingface transformers backbone
# run clip_adapter
bash scripts/clip_finetune/clip_adapter_hf.sh caltech101 vit_b16 0
# run clip_adapter and do val/train acc cal every 10 epoch
bash scripts/clip_finetune/clip_adapter_hf.sh caltech101 vit_b16 10
# run clip_full_finetune
bash scripts/clip_finetune/clip_fullfinetune_hf.sh caltech101 vit_b16 0
# run clip_bias
bash scripts/clip_finetune/clip_bias_hf.sh caltech101 vit_b16 0
# run clip_prompt with 1 prompt length and use Deep VPT
bash scripts/clip_finetune/clip_prompt_hf.sh caltech101 vit_b16 1 True 0
# run clip_prompt with 2 prompt length and don't use Deep VPT
bash scripts/clip_finetune/clip_prompt_hf.sh caltech101 vit_b16 2 False 0
# run clip_prompt with 1 prompt length and use Deep VPT using mini-imagenet dataset
bash scripts/clip_finetune/clip_prompt_hf.sh mini_imagenet vit_b16 1 True 0
# run clip_adapter using flickr30k dataset
bash scripts/clip_finetune/clip_adapter_hf.sh flickr30k vit_b16 0

# checkpoint will save to output/$METHOD/$MODEL/$DATASET
# you can set `export CLIP_DEBUG=1` to remove checkpoint
```

# config yaml for clip_bias

```python
we use yaml to config param.
e.g.       ./configs/clip_finetune/vit_b16_bias_example.yaml
BIAS_TERMS:             which layer's bias you want to tune, default to tune all bias layer, in this config we tune the layer with attn or mlp in name
BIAS_TERMS_EXCLUDE      which layer's bias you don't need to tune, in this config we don't tune text_encoder
```

## code structure

```python
./scripts/clip_finetune contains the scripts we use to run
./trainers contains model related code
```

### Run on single A770

```bash
# run with A770
# run with huggingface transformers backbone
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
bash scripts/clip_finetune/clip_adapter_hf.sh caltech101 vit_b16 0 XPU
bash scripts/clip_finetune/clip_fullfinetune_hf.sh caltech101 vit_b16 0 XPU
bash scripts/clip_finetune/clip_bias_hf.sh caltech101 vit_b16 0 XPU
bash scripts/clip_finetune/clip_prompt_hf.sh caltech101 vit_b16 1 True 0 XPU

```

### Run on A770 with DDP

```python
export NEOReadDebugKeys=1
export DisableScratchPages=0
export CCL_ATL_TRANSPORT=ofi

bash scripts/clip_finetune/clip_adapter_hf_muti.sh caltech101 vit_b16 0 XPU
```

# use optuna to automatic get the best param

You can use optuna(https://github.com/optuna/optuna) to automatic tune the hyperparameter.
We only support turn bs and lr.
You can set the bs and lr in yaml, such as ./configs/clip_finetune/vit_b16_opt.yaml

```python
# turn on optuna in A100
bash scripts/clip_finetune/clip_adapter_hf_opt.sh caltech101 vit_b16 0 cuda 1
# turn on optuna in A770
bash scripts/clip_finetune/clip_adapter_hf_opt.sh caltech101 vit_b16 0 XPU 1
```

## Performance of different finetune methods on Caltech-101

| Finetune method     | epochs | batch size | LR   | Consume GPU Memory | train | test | Acc   | Consume Time |
| ------------------- | ------ | ---------- | ---- | ------------------ | ----- | ---- | ----- | ------------ |
| Full Finetune       | 200    | 32         | 1e-5 | 11422 MB           | 4128  | 2465 | 96.9% | 311.75 Min   |
| clip Bias           | 200    | 32         | 2e-2 | 7640 MB            | 4128  | 2465 | 96.4% | 224.03 Min   |
| clip VPT Deep       | 200    | 32         | 2e-2 | 4866 MB            | 4128  | 2465 | 96.7% | 172 Min      |
| clip Adapter        | 200    | 32         | 2e-2 | 1584 MB            | 4128  | 2465 | 96.3% | 95.46 Min    |
| Full Finetune + ABS | 200    | 32         | 2e-2 | 8158 MB            | 4128  | 2465 | 96.5% | 110.08 Min   |

# problem

```bash
# if you hit below problem with DDP,
#     [1] [1728626096.870265672] DUT7113ATSM:rank1.python: Reading from remote process' memory failed. Disabling CMA support
#     [1] DUT7113ATSM:rank1: Assertion failure at psm3/ptl_am/ptl.c:210: nbytes == req->req_data.recv_msglen
# You can run
    echo 0 >> /proc/sys/kernel/yama/ptrace_scope
```
