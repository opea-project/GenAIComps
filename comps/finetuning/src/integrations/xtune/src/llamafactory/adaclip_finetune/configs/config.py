# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys

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


parser = argparse.ArgumentParser(description="PyTorch implementation of Transformer Video Retrieval")

parser.add_argument("--dataset", default="", help="dataset name")
parser.add_argument("--train_annot", default="", help="json file containing training video annotations")
parser.add_argument("--val_annot", default="", help="json file containing validation video annotations")
parser.add_argument("--test_annot", default="", help="json file containing test video annotations")
parser.add_argument("--frames_dir", default="", type=str, help="path to video frames")
parser.add_argument("--output_dir", type=str, default="output", help="dir to store model checkpoints & training meta.")
parser.add_argument("--tensorboard_dir", type=str, default="tensorboard", help="dir to store tensorboard")
parser.add_argument("--config", help="config file path")
parser.add_argument("--xpu", action="store_true", default=False, help="whether use XPU")
# ========================= Model Configs ==========================
parser.add_argument(
    "--policy_backbone",
    default="mobilenet_v3_large",
    type=str,
    choices=["raw", "clip", "frozen_clip", "resnet50", "mobilenet_v2", "mobilenet_v3_large"],
    help="type of visual backbone for policy",
)
parser.add_argument(
    "--clip_backbone",
    default="ViT-B/32",
    type=str,
    choices=["ViT-B/32", "ViT-B/16"],
    help="type of visual backbone for CLIP",
)
parser.add_argument(
    "--rnn", default="transformer", type=str, choices=["lstm", "bilstm", "transformer"], help="type of RNN backbone"
)
parser.add_argument("--hidden_dim", default=512, type=int, help="RNN hidden dim")
parser.add_argument("--mlp_hidden_dim", default=1024, type=int, help="MLP hidden dim")
parser.add_argument(
    "--mlp_type",
    type=str,
    default="mlp",
    choices=["mlp", "fc"],
    help="type of linear model to use before gumbel softmax",
)
parser.add_argument(
    "--rescale_size", default=56, type=int, help="Rescale size for using pixel differences (no CNN backbone)"
)
parser.add_argument("--no_policy", action="store_false", dest="use_policy", help="no policy network")
parser.add_argument("--no_rnn", action="store_false", dest="use_rnn", help="no rnn to encode visual features")
parser.add_argument(
    "--sim_header",
    default="meanP",
    choices=["meanP", "seqTransf", "transformer"],
    help="similarity header to aggregate frame features",
)
parser.add_argument(
    "--word_agg", default=None, choices=["mlp", "transformer"], help="method to learn weights for word aggregation"
)
parser.add_argument(
    "--word_agg_temp", type=float, default=1, help="temperature parameter used in word aggregation softmax"
)
parser.add_argument(
    "--frame_agg",
    default=None,
    choices=["mlp", "transformer", "qscore"],
    help="method to learn weights for frame aggregation",
)
parser.add_argument(
    "--frame_agg_temp", type=float, default=1, help="temperature parameter used in frame aggregation softmax"
)
parser.add_argument("--freeze_layer_num", type=int, default=0, help="layer NO. of CLIP need to freeze")
parser.add_argument("--top_k", default=16, type=int, help="select top K frames in a video")
parser.add_argument("--reuse_scores", action="store_true", help="reuse frame selection scores for aggregation")

# ========================= Preprocessing Configs ==========================
parser.add_argument("--max_txt_len", type=int, default=20, help="max text #tokens ")
parser.add_argument(
    "--concat_captions",
    default="concat",
    choices=["concat", "indep-one", "indep-all"],
    help="concatenate video captions",
)
parser.add_argument(
    "--max_img_size", type=int, default=224, help="max image longer side size, shorter side will be padded with zeros"
)
parser.add_argument("--num_frm", type=int, default=2, help="#frames to use per video")
parser.add_argument(
    "--num_frm_subset", type=int, default=0, help="ablation study: number of frames to sample from num_frm frames"
)
parser.add_argument(
    "--sampling", type=str, default="uniform", choices=["uniform", "random"], help="how to sample frames"
)
parser.add_argument("--img_tmpl", default="image_{:05d}.jpg", type=str, help="frame filename pattern")
parser.add_argument("--segment_sampling", action="store_true", help="sample mid frames from N segments")

# ========================= Learning Configs ==========================
parser.add_argument("--batch_size", default=32, type=int, help="single-GPU batch size.")
parser.add_argument("--val_batch_size", default=500, type=int, help="single-GPU batch size.")
parser.add_argument("--num_epochs", default=30, type=int, help="total # of training epochs.")
parser.add_argument("--warmup_epochs", default=0, type=int, help="total # of warm up epochs for annealing K.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="initial learning rate.")
parser.add_argument("--coef_lr", type=float, default=1e-3, help="lr multiplier for clip branch")
parser.add_argument("--no_warmup", action="store_true", help="do not perform cosine warmup LR")
parser.add_argument(
    "--warmup_proportion",
    default=0.1,
    type=float,
    help="proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training",
)
parser.add_argument("--optim", default="bertadam", choices=["bertadam", "adamw"], help="optimizer")
parser.add_argument("--betas", default=[0.9, 0.98], nargs=2, help="beta for adam optimizer")
parser.add_argument("--weight_decay", default=0.2, type=float, help="weight decay (L2) regularization")
parser.add_argument("--grad_norm", default=1.0, type=float, help="gradient clipping (-1 for no clipping)")
parser.add_argument(
    "--freeze_cnn", action="store_true", help="freeze CNN by setting the requires_grad=False for CNN parameters."
)
parser.add_argument(
    "--freeze_clip", action="store_true", help="freeze CLIP by setting the requires_grad=False for CLIP parameters."
)
parser.add_argument("--init_tau", default=5.0, type=float, help="annealing init temperature")
parser.add_argument("--min_tau", default=0.5, type=float, help="min temperature to anneal to")
parser.add_argument("--exp_decay_factor", default=0.045, type=float, help="exp decay factor per epoch")

# ========================= Runtime Configs ==========================
parser.add_argument("--resume", type=str, help="path to latest checkpoint")
parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
parser.add_argument("--num_workers", type=int, default=16, help="#workers for data loading")
parser.add_argument("--do_inference", action="store_true", help="perform inference run")
parser.add_argument("--pin_mem", action="store_true", help="pin memory")
parser.add_argument("--debug", action="store_true", help="debug mode. Log extra information")
parser.add_argument(
    "--data_subset", default=0, type=int, help="debug mode. Use only this number of samples for training and testing"
)
parser.add_argument("--no_output", action="store_true", help="do not save model and logs")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="# of updates steps to accumulate before performing a backward/update pass."
    "Used to simulate larger batch size training. The simulated batch size "
    "is batch_size * gradient_accumulation_steps for a single GPU.",
)
parser.add_argument("--n_display", type=int, default=20, help="Information display frequency")
parser.add_argument("--save_last", action="store_true", help="save last epoch's checkpoint")
parser.add_argument("--optuna", action="store_true", default=False, help="whether use optuna to tune parameters")
parser.add_argument("--n_trials", type=int, default=100, help="Tune times of optuna")
parser.add_argument(
    "--do_training_af_optuna",
    action="store_true",
    help="whether use optuna tuned parameters to train after optuna tuning",
)

# ========================= Distributed Configs ==========================
parser.add_argument("--world_size", default=1, type=int, help="distributed training")
parser.add_argument("--local_rank", default=0, type=int, help="distributed training")
parser.add_argument("--rank", default=0, type=int, help="distributed training")

# ========================= Profiling Configs ==========================
parser.add_argument("--prof_type", default="forward-backward", type=str, help="profiling type")
