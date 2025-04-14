# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
from functools import partial
from typing import List, Optional

import optuna

from ...adaclip_finetune import *
from ...adaclip_finetune.train import *

adaclip_parser = argparse.ArgumentParser(
    prog="adaclip", description="PyTorch implementation of Transformer Video Retrieval"
)

adaclip_parser.add_argument("--dataset", default="", help="dataset name")
adaclip_parser.add_argument("--train_annot", default="", help="json file containing training video annotations")
adaclip_parser.add_argument("--val_annot", default="", help="json file containing validation video annotations")
adaclip_parser.add_argument("--test_annot", default="", help="json file containing test video annotations")
adaclip_parser.add_argument("--frames_dir", default="", type=str, help="path to video frames")
adaclip_parser.add_argument(
    "--output_dir", type=str, default="output", help="dir to store model checkpoints & training meta."
)
adaclip_parser.add_argument("--tensorboard_dir", type=str, default="tensorboard", help="dir to store tensorboard")
adaclip_parser.add_argument("--config", help="config file path")
adaclip_parser.add_argument("--xpu", action="store_true", default=False, help="whether use XPU")
# ========================= Model Configs ==========================
adaclip_parser.add_argument(
    "--policy_backbone",
    default="mobilenet_v3_large",
    type=str,
    choices=["raw", "clip", "frozen_clip", "resnet50", "mobilenet_v2", "mobilenet_v3_large"],
    help="type of visual backbone for policy",
)
adaclip_parser.add_argument(
    "--clip_backbone",
    default="ViT-B/32",
    type=str,
    choices=["ViT-B/32", "ViT-B/16"],
    help="type of visual backbone for CLIP",
)
adaclip_parser.add_argument(
    "--rnn", default="transformer", type=str, choices=["lstm", "bilstm", "transformer"], help="type of RNN backbone"
)
adaclip_parser.add_argument("--hidden_dim", default=512, type=int, help="RNN hidden dim")
adaclip_parser.add_argument("--mlp_hidden_dim", default=1024, type=int, help="MLP hidden dim")
adaclip_parser.add_argument(
    "--mlp_type",
    type=str,
    default="mlp",
    choices=["mlp", "fc"],
    help="type of linear model to use before gumbel softmax",
)
adaclip_parser.add_argument(
    "--rescale_size", default=56, type=int, help="Rescale size for using pixel differences (no CNN backbone)"
)
adaclip_parser.add_argument("--no_policy", action="store_false", dest="use_policy", help="no policy network")
adaclip_parser.add_argument("--no_rnn", action="store_false", dest="use_rnn", help="no rnn to encode visual features")
adaclip_parser.add_argument(
    "--sim_header",
    default="meanP",
    choices=["meanP", "seqTransf", "transformer"],
    help="similarity header to aggregate frame features",
)
adaclip_parser.add_argument(
    "--word_agg", default=None, choices=["mlp", "transformer"], help="method to learn weights for word aggregation"
)
adaclip_parser.add_argument(
    "--word_agg_temp", type=float, default=1, help="temperature parameter used in word aggregation softmax"
)
adaclip_parser.add_argument(
    "--frame_agg",
    default=None,
    choices=["mlp", "transformer", "qscore"],
    help="method to learn weights for frame aggregation",
)
adaclip_parser.add_argument(
    "--frame_agg_temp", type=float, default=1, help="temperature parameter used in frame aggregation softmax"
)
adaclip_parser.add_argument("--freeze_layer_num", type=int, default=0, help="layer NO. of CLIP need to freeze")
adaclip_parser.add_argument("--top_k", default=16, type=int, help="select top K frames in a video")
adaclip_parser.add_argument("--reuse_scores", action="store_true", help="reuse frame selection scores for aggregation")

# ========================= Preprocessing Configs ==========================
adaclip_parser.add_argument("--max_txt_len", type=int, default=20, help="max text #tokens ")
adaclip_parser.add_argument(
    "--concat_captions",
    default="concat",
    choices=["concat", "indep-one", "indep-all"],
    help="concatenate video captions",
)
adaclip_parser.add_argument(
    "--max_img_size", type=int, default=224, help="max image longer side size, shorter side will be padded with zeros"
)
adaclip_parser.add_argument("--num_frm", type=int, default=2, help="#frames to use per video")
adaclip_parser.add_argument(
    "--num_frm_subset", type=int, default=0, help="ablation study: number of frames to sample from num_frm frames"
)
adaclip_parser.add_argument(
    "--sampling", type=str, default="uniform", choices=["uniform", "random"], help="how to sample frames"
)
adaclip_parser.add_argument("--img_tmpl", default="image_{:05d}.jpg", type=str, help="frame filename pattern")
adaclip_parser.add_argument("--segment_sampling", action="store_true", help="sample mid frames from N segments")

# ========================= Learning Configs ==========================
adaclip_parser.add_argument("--batch_size", default=32, type=int, help="single-GPU batch size.")
adaclip_parser.add_argument("--val_batch_size", default=500, type=int, help="single-GPU batch size.")
adaclip_parser.add_argument("--num_epochs", default=30, type=int, help="total # of training epochs.")
adaclip_parser.add_argument("--warmup_epochs", default=0, type=int, help="total # of warm up epochs for annealing K.")
adaclip_parser.add_argument("--learning_rate", default=1e-4, type=float, help="initial learning rate.")
adaclip_parser.add_argument("--coef_lr", type=float, default=1e-3, help="lr multiplier for clip branch")
adaclip_parser.add_argument("--no_warmup", action="store_true", help="do not perform cosine warmup LR")
adaclip_parser.add_argument(
    "--warmup_proportion",
    default=0.1,
    type=float,
    help="proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training",
)
adaclip_parser.add_argument("--optim", default="bertadam", choices=["bertadam", "adamw"], help="optimizer")
adaclip_parser.add_argument("--betas", default=[0.9, 0.98], nargs=2, help="beta for adam optimizer")
adaclip_parser.add_argument("--weight_decay", default=0.2, type=float, help="weight decay (L2) regularization")
adaclip_parser.add_argument("--grad_norm", default=1.0, type=float, help="gradient clipping (-1 for no clipping)")
adaclip_parser.add_argument(
    "--freeze_cnn", action="store_true", help="freeze CNN by setting the requires_grad=False for CNN parameters."
)
adaclip_parser.add_argument(
    "--freeze_clip", action="store_true", help="freeze CLIP by setting the requires_grad=False for CLIP parameters."
)
adaclip_parser.add_argument("--init_tau", default=5.0, type=float, help="annealing init temperature")
adaclip_parser.add_argument("--min_tau", default=0.5, type=float, help="min temperature to anneal to")
adaclip_parser.add_argument("--exp_decay_factor", default=0.045, type=float, help="exp decay factor per epoch")

# ========================= Runtime Configs ==========================
adaclip_parser.add_argument("--resume", type=str, help="path to latest checkpoint")
adaclip_parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
adaclip_parser.add_argument("--num_workers", type=int, default=16, help="#workers for data loading")
adaclip_parser.add_argument("--do_inference", action="store_true", help="perform inference run")
adaclip_parser.add_argument("--pin_mem", action="store_true", help="pin memory")
adaclip_parser.add_argument("--debug", action="store_true", help="debug mode. Log extra information")
adaclip_parser.add_argument(
    "--data_subset", default=0, type=int, help="debug mode. Use only this number of samples for training and testing"
)
adaclip_parser.add_argument("--no_output", action="store_true", help="do not save model and logs")
adaclip_parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="# of updates steps to accumulate before performing a backward/update pass."
    "Used to simulate larger batch size training. The simulated batch size "
    "is batch_size * gradient_accumulation_steps for a single GPU.",
)
adaclip_parser.add_argument("--n_display", type=int, default=20, help="Information display frequency")
adaclip_parser.add_argument("--save_last", action="store_true", help="save last epoch's checkpoint")
adaclip_parser.add_argument(
    "--optuna", action="store_true", default=False, help="whether use optuna to tune parameters"
)
adaclip_parser.add_argument("--n_trials", type=int, default=100, help="Tune times of optuna")
adaclip_parser.add_argument(
    "--do_training_af_optuna",
    action="store_true",
    help="whether use optuna tuned parameters to train after optuna tuning",
)

# ========================= Distributed Configs ==========================
adaclip_parser.add_argument("--world_size", default=1, type=int, help="distributed training")
adaclip_parser.add_argument("--local_rank", default=0, type=int, help="distributed training")
adaclip_parser.add_argument("--rank", default=0, type=int, help="distributed training")

# ========================= Profiling Configs ==========================
adaclip_parser.add_argument("--prof_type", default="forward-backward", type=str, help="profiling type")
from easydict import EasyDict as edict


def parse_with_config(parsed_args):
    """This function will set args based on the input config file.

    it only overwrites unset parameters,
    i.e., these parameters not set from user command line input
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    # del args.config
    return args


# UI点击执行后，会走到这个函数
def run_sft_adaclip(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    adaclip_args: "AdaclipArguments",
    optuna_args: "OptunaArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):

    # 初始化原本adaclip的参数
    parsed_args, _ = adaclip_parser.parse_known_args()
    # 将config文件传到原本adaclip的参数里
    parsed_args.config = adaclip_args.config
    # 根据config文件修改参数
    args = parse_with_config(parsed_args)
    # 根据UI传入的参数修改
    args.resume = adaclip_args.resume
    args.frames_dir = adaclip_args.frames_dir + "/" + data_args.dataset[0] + "/frames"
    args.output_dir = training_args.output_dir
    args.top_k = adaclip_args.adaclip_top_k
    args.batch_size = adaclip_args.adaclip_batch_size
    args.xpu = adaclip_args.adaclip_xpu
    args.freeze_cnn = adaclip_args.freeze_cnn
    args.frame_agg = adaclip_args.frame_agg
    args.num_epochs = int(training_args.num_train_epochs)
    dataset_config_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/adaclip_finetune/"
    args.train_annot = dataset_config_path + args.train_annot
    args.val_annot = dataset_config_path + args.val_annot
    args.test_annot = dataset_config_path + args.test_annot
    # if "optuna_cfg" in args:
    #     args.optuna = True
    if optuna_args.optuna:
        args.optuna = True
        args.optuna_cfg = edict()
        args.optuna_cfg.n_trials = optuna_args.n_trials
        args.optuna_cfg.n_warmup_steps = optuna_args.n_warmup_steps
        args.optuna_cfg.sampler = edict()
        args.optuna_cfg.sampler.name = optuna_args.sampler
        args.optuna_cfg.opt_params = edict(json.loads(optuna_args.opt_params))
    else:
        args.optuna = False
    # 获取UI获得的参数
    args.save_last = os.getenv("SAVE_LAST", False)
    print("adaclip_args", args)
    if args.optuna:
        # do optuna to tune parameters in the config file
        storage_name = "sqlite:///adaclip_optuna.db"
        objective_fn = partial(objective, cfg=args)
        sampler = get_sampler(args)
        # set up and run the optuna study
        study = optuna.create_study(
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.optuna_cfg.n_warmup_steps),
            direction="maximize",
            study_name="adaclip_bitfit_coef_lr00205_weight_decay00105_n_trials30",
            storage=storage_name,
            load_if_exists=True,
        )
        study.optimize(objective_fn, n_trials=args.optuna_cfg.n_trials, gc_after_trial=True)
        best_hyperparameters = study.best_trial.params
        print("best_hyperparameters:", best_hyperparameters)
        if args.do_training_af_optuna:
            for param_name, param_data in best_hyperparameters.items():
                setattr(args, param_name, param_data)
            # using optuna tuned parameters to do training.
            train(args)
    else:
        train(args)
