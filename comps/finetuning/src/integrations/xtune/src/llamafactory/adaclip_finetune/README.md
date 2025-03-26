# AdaCLIP-Finetune

This repo is the finetune implementation for the paper "AdaCLIP: Towards Pragmatic Multimodal Video Retrieval"

![arch](images/adaclip_design.jpg)

Incorporating large image-text foundation models such as CLIP has
substantially improved the performance of the multimodal video
retrieval task. However, how to practically sample the frames from
a video and aggregate the frame features into a video representation
is still an open research question. In particular, real-world
deployment scenarios, such as embodiment within consumer electronics
or cloud-based inference pipelines, require two key facets of
retrieval (representation building and search) to be computationally
light and fast. In this paper, we propose AdaCLIP, a computationand
latency-aware system for pragmatic multimodal video retrieval.
AdaCLIP consists of a _learning-based frame selection module_ to select
informative frames and a _query-independent frame aggregation
module_ to obtain strong video representations from the frame features.
Specifically, in the frame selection module, we introduce a
differentiable _Hard-Top-k_ algorithm to sample a subset of the frames
while optimizing the performance of the video retrieval task in an
end-to-end manner. Moreover, to be latency-aware, we also propose
a query-independent lightweight approach, _MLP-Score_, to aggregate
the frame features into the video representation, which offers
up to 142x speedup on GPU and 822x speedup on CPU in similarity
search time compared to query-dependent matching methods.
Experimental results on several popular video retrieval datasets
confirm the effectiveness of AdaCLIP.

# Prerequisites

- Linux (Ubuntu 22.04.1 or later is recommended)
- Python 3.10
- Packages:
  - ffmpeg (`$sudo apt-get install ffmpeg`)
- Datasets: [ActivityNet Dense Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/), [MSRVTT](http://ms-multimedia-challenge.com/2017/dataset), [DiDeMo](https://github.com/LisaAnne/LocalizingMoments)

# How to Install

## Install on NVIDIA

Create a conda environment and install the appropriate packages:

```sh
conda activate adaclip_py310_nv
conda create -n adaclip_py310_nv python=3.10 -y
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.7 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## Install on Arc A770

### Install Driver for Arc 770

please fololow [Install Dependency](../../../doc/install_dependency.md) to install Driver for Arc 770

### Install oneapi

You can refer to https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

```sh
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/dfc4a434-838c-4450-a6fe-2fa903b75aa7/intel-oneapi-base-toolkit-2025.0.1.46_offline.sh
sudo sh ./intel-oneapi-base-toolkit-2025.0.1.46_offline.sh -a --silent --cli --eula accept
```

### Create a conda environment install IPEX and other lib

#### Create a conda environment

```sh
conda create -n adaclip_py310 python=3.10 -y
conda activate adaclip_py310
```

#### Install ipex

You can refer to https://github.com/intel/intel-extension-for-pytorch

```sh
python -m pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi torchaudio==2.5.1+cxx11.abi intel-extension-for-pytorch==2.5.10+xpu oneccl_bind_pt==2.5.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

Check xpu:

```sh
python
import torch
import intel_extension_for_pytorch
torch.xpu.device_count()
```

#### Install requirements

```sh
pip install -r requirements.txt
```

# Prepare Datasets

## Datasets

We mainly use `ActivityNet` to do finetune, you can also use other datasets.

The training data information is located in the directories `src/llamafactory/adaclip_finetune/annots-finetune` and `src/llamafactory/adaclip_finetune/annots/`. To change the finetuning and validation datasets, you can modify the `dataset`, `train_annot`, and `val_annot` paths in the finetune configurations found under `src/llamafactory/adaclip_finetune/cfgs`.

We primarily use the `src/llamafactory/adaclip_finetune/annots-finetune/activitynet/finetune-5000.json` file for fine-tuning.

For validation during the fine-tuning process, we utilize the `src/llamafactory/adaclip_finetune/annots/activitynet/val.json` file. The validation is performed at the end of each epoch. The validation indicator is best recall(top1 and top5)

### ActivityNet

Download the videos from the [official website](http://activity-net.org/download.html). The authors have made the videos available on Google and Baidu drives.

### MSRVTT

The videos are shared by [Frozen in Time](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt):

```
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### DiDeMo

The videos can be downloaded from [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments).

## Frame Extraction

Run `utils/frame_extraction.py` after having downloaded the dataset videos and annotations from the website. Make sure that all the videos are in the same directory (no sub-directories allowed).

```
python utils/frame_extraction.py /path/to/videos /path/to/frames --parallel
```

# Implemented finetune methods

We have implemented the BitFit and IBS fine-tuning methods.

To fine-tune using different methods, you can utilize the corresponding configuration files located under `src/llamafactory/adaclip_finetune/cfgs/peft`.
For a more detailed guide, please refer to [How to Finetune](#how-to-finetune) section.

## BitFit & SSF

[BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models](https://aclanthology.org/2022.acl-short.1)  
[Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning](https://papers.neurips.cc/paper_files/paper/2022/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html)  
[Revisiting Batch Normalization For Practical Domain Adaptation](https://openreview.net/forum?id=Hk6dkJQFx)

Example

```json
    ...
    "peft": {
        "method": "bitfit",
        "config": {
            "keep_module_keywords": [
                "ln_post",
                "visual.proj",
                "ln_final",
                "text_projection",
                "logit_scale"
            ]
        }
    }
    ...
```

Config path: `src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-bitfit-5k.json`

**TODO**: check if the naive recusive monkey patch has problems.

## Importance Based Selection (IBS)

Select partial layers for finetune based on the parameter updates after training a given steps/epochs. The metric for importance can be either the l2 norm of param updates or angle based, which is introduced in the following paper:

[Angle-based Search Space Shrinking for Neural Architecture Search](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3155_ECCV_2020_paper.php)

Example

```json
    ...
    "peft": {
        "method": "ibs",
        "config": {
            "pre_batch_size": 8,
            "num_pre_epochs": 2,
            "retain_ratio": 0.05,
            "metric": "l2norm",
            "normalization": true,
            "keep_module_keywords": [
                "ln_post",
                "visual.proj",
                "ln_final",
                "text_projection",
                "logit_scale"
            ]
        }
    }
    ...
```

Config path:
`src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-ibs-r005-5k.json`
`src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-ibs-r010-5k.json`

## Performance of different finetune methods

| Finetune method | # frames | Top-k | epochs | batch size | LR(Main/CLIP) | % params | # train | # test | T2V: R1/R5 | V2T: R1/R5 |
| --------------- | -------- | ----- | ------ | ---------- | ------------- | -------- | ------- | ------ | ---------- | ---------- |
| Full Finetune   | 32       | 16    | 30     | 16         | 1e-4/1e-7     | 100      | 5000    | 4917   | 37.9/68.2  | 38.5/69.3  |
| IBS-G Finetune  | 32       | 16    | 30     | 16         | 1e-4/1e-7     | 8.314    | 5000    | 4917   | 36.8/67.4  | 38.4/68.3  |
| BitFit Finetune | 32       | 16    | 30     | 16         | 1e-4/2e-5     | 0.516    | 5000    | 4917   | 36.3/66.2  | 37.7/68.4  |

# How to Finetune

You can finetune AdaCLIP by using `src/llamafactory/adaclip_finetune/train.py` and configs under `src/llamafactory/adaclip_finetune/cfgs`.

The bitfit and ibs configs are under `src/llamafactory/adaclip_finetune/cfgs/peft`.

The full finetune config is under `src/llamafactory/adaclip_finetune/cfgs/finetune`.

You can modify the information in config jsons to meet your requirements.

## Finetune on NVIDIA

```sh
cd src/llamafactory/adaclip_finetune
```

Finetune AdaCLIP with bitfit

```sh
python  train.py --config src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-bitfit-5k.json --frames_dir  /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pre-train/model --batch_size 8
```

Finetune AdaCLIP with ibs

```sh
python  train.py --config src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-ibs-r005-5k.json (or activitynet-ibs-r010-5k.json) --frames_dir  /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pre-train/model --batch_size 8
```

Full finetune

```sh
python  train.py --config src/llamafactory/adaclip_finetune/cfgs/finetune/activitynet-finetune-5000-c-32.json --frames_dir  /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pretrain/model --batch_size 8
```

## Finetune on Arc A770

Currently only single card finetune is supported, you can specify the XPU with the following command:

```sh
export ZE_AFFINITY_MASK=the_card_number
```

Enter the AdaCLIP folder:

```sh
cd src/llamafactory/adaclip_finetune
```

Finetune AdaCLIP with bitfit

```sh
python  train.py --config src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-bitfit-5k.json --frames_dir  /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pretrain/model --xpu --batch_size 8
```

Finetune AdaCLIP with ibs

```sh
python  train.py --config src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-ibs-r005-5k.json (or activitynet-ibs-r010-5k.json) --frames_dir  /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pretrain/model --xpu --batch_size 8
```

Full finetune

```sh
python  train.py --config src/llamafactory/adaclip_finetune/cfgs/finetune/activitynet-finetune-5000-c-32.json --frames_dir  /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pretrain/model --xpu --batch_size 8
```

The finetune output will located in `src/llamafactory/adaclip_finetune/output`

# Use optuna to automatic get the best param

You can enable optuna to automatic get the best param by adding `optuna_cfg` configs to config files like:

```sh
    "optuna_cfg": {
        "n_trials": 30,
        "n_warmup_steps":10,
        "sampler": {
            "name": "TPESampler"
        },
        "opt_params": {
            "coef_lr": {
                "range": [0.02,0.5],
                "log": false
            },
            "weight_decay": {
                "range": [0.01,0.5],
                "log": false
            }
        }
    }
```

The config example is: `src/llamafactory/adaclip_finetune/cfgs/peft/activitynet-bitfit-5k-optuna.json`
|Config name|Description|
|:--|:--|
|n_trials|The max number of trials. Must be set to an integer.|
|n_warmup_steps|The pruning is disabled until the trial exceeds the given number of step(epochs). Note that this feature assumes that step starts at zero.
|sampler|Choose samplers which optuna uses. now support `TPESampler`,`CmaEsSampler` and `GPSampler`.|
|opt_params|The parameters you want to optimize.|

| Configs of opt_params | Description                                                                                                                                                                                        |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| range                 | The min and max value of the parameter.                                                                                                                                                            |
| log                   | A flag to sample the value from the log domain or not. If log is true, the value is sampled from the range in the log domain. Otherwise, the value is sampled from the range in the linear domain. |

If you want to continue train models with the best parameters after optuna optimization, add `--do_training_af_optuna` in your command line.

Command example:

```sh
cd src/llamafactory/adaclip_finetune/train.py
python train.py --config ./cfgs/peft/activitynet-bitfit-5k-c-32_optuna.json --frames_dir /path/to/ActivityNet/frames --top_k 16 --freeze_cnn --frame_agg mlp --resume /path/to/pre-train/model --xpu --batch_size 8
```

## Visualization

You can review optuna tuning results by:

```sh
sudo ufw allow 8084
optuna-dashboard --host 0.0.0.0 --port 8084 sqlite:///optuna.db
```

Open in the website:

```
http://<serverIP>:8084/dashboard
```

You can see finetune curves for different parameters and other infornations in the website.
![alt text](optuna.png)
