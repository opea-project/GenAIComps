# The core features for clip finetune tool:

Below method can run on Classification task and Image to Text task

<table width="100%">
    <tr>
        <td align="center" colspan="1"><strong>Method</strong></td>
        <td align="center" colspan="1"><strong>Detail Description</strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong>Full Finetune</strong></td>
        <td align="center" colspan="1"><strong>1. Default update all parameters<br>
        2. Enable <a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3155_ECCV_2020_paper.php">Angle-Based Selection</a>(base on the weight angle to determine which layer to update)<br>
        3. Enable <a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3155_ECCV_2020_paper.php">Angle-Based Selection</a>(base on the weight angle to determine which layer to update)<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong>Partial Finetuning - bias</strong></td>
        <td align="center" colspan="1"><strong>1. Default update all bias parameters<br>
        2. Allow users to customize which layers participate in training and which ones do not<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong><a href="https://arxiv.org/abs/2203.12119">Prompt Tuning</a></strong></td>
        <td align="center" colspan="1"><strong>adding prompt embedding layer at the head of model or at the inputs of every layer and only train these layers<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong><a href="https://arxiv.org/pdf/2110.04544">Adapter Tuning</a></strong></td>
        <td align="center" colspan="1"><strong>adding adapter network at the end of encoder and only train this network<br></strong></td>       
    <tr>
    <tr>
        <td align="center" colspan="1"><strong>Training free - <a href="https://arxiv.org/pdf/2207.09519">Tip Adapter</a></strong></td>
        <td align="center" colspan="1"><strong>1. finetune CLIP model without any training or with few epochs learning<br>
        2. Added fixed cache size to reduce memory and enable experience sharing across different datasets</strong></td>       
    <tr>
</table>

# How to config features for clip finetune tool:

## Basic yaml for vit_b16.yaml

see this in src/llamafactory/clip_finetune/configs/clip_finetune/vit_b16.yaml

```bash
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 32
  TEST:
    BATCH_SIZE: 100
  NUM_WORKERS: 4

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "sgd"
  LR: 0.02
  MAX_EPOCH: 50
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5

TRAIN:
  PRINT_FREQ: 1             # print acc after $PRINT_FREQ iteration

MODEL:
  BACKBONE:
    NAME: "ViT-B/16"
```

## Angle-Based Selection for full-finetune

Add below line in src/llamafactory/clip_finetune/configs/clip_finetune/vit_b16.yaml

```bash
MODEL:
  ABS: True
  ABS_TOP: True                     # True: select top ABS_KEEP layer  False: select bottom ABS_KEEP layer
  ABS_GROUP: True                   # True: select top ABS_KEEP layer in each group False: select bottom ABS_KEEP layer
  ABS_GROUP_NAME: ["k_proj", "v_proj", "q_proj"]    # How to divide layer into GTOUP, this means divide layers into 4 group. Each layer has k_proj in its name will into group 0, v_proj into group1, q_proj into group 2, other into group 3
  ABS_KEEP: 5                       # keep layer number
  BACKBONE:
    NAME: "ViT-B/16"
```

## customize trained layer for partial-finetune

Add below line in src/llamafactory/clip_finetune/configs/clip_finetune/vit_b16.yaml

```bash
MODEL:
  BACKBONE:
    NAME: "ViT-B/16"
BIAS:
  BIAS_TERMS: ["layer_norm", "layernorm"]   # which layer you want to train
  BIAS_TERMS_EXCLUDE: ["layernorm"]         # which layer you don't want to train
```

## fixed cache size for tip-adapter

Add below line in src/llamafactory/clip_finetune/configs/clip_finetune/vit_b16.yaml

```bash
TRAINER:
  TIP:
    LOAD_CACHE: True                    # whether to use cache data trained with tip-adapter before
    beta: 1.0                           # hyper param in origin paper
    alpha: 3.0                          # hyper param in origin paper
    AUGMENT_EPOCH: 10                   # train cache epoch
    search_best: True                   # whether to search the best beta and alpha
    NEW: False                          # whether to use fixed cache size. True: all dataset cache will merge into one tensor [100, hidden_size]   False: each dataset will has it's own cache [num_dataset * 100, hidden_size]
    NEW_DATASET: False                  # Whether to train this dataset from scratch
```
