#!/bin/bash

# custom config

TRAINER=CLIP_Bias
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
ACCEP=$3
DEVICE=$4
if [ -z $ACCEP ]; then
    ACCEP=0
fi
if [ -z $DEVICE ]; then
    DEVICE="cuda"
    device=0
fi
if [ $DEVICE = "XPU" ]; then
    device=1
fi
CUDA_VISIBLE_DEVICES=3 python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/clip_finetune/${CFG}_bias.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--xpu $device \
TRAINER.COOP.ACC $ACCEP \
TRAINER.COOP.N_CTX 16 \
TRAINER.COOP.CSC True \
TRAINER.COOP.CLASS_TOKEN_POSITION end \
DATASET.NUM_SHOTS 0