# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil

# custom
import optuna
import torch
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger
from optuna.trial import TrialState


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COOP.N_PLN = 4  # initialization prompt length for clip prompt
    cfg.TRAINER.COOP.N_PDT = 0.0  # initialization prompt_dropout for clip prompt
    cfg.TRAINER.COOP.PMT_DEEP = True  # initialization prompt_dropout for clip prompt
    cfg.TRAINER.COOP.ACC = 0
    cfg.TRAINER.COOP.XPU = False
    cfg.TRAINER.COOP.XPU_ID = "xpu:0"
    cfg.TRAINER.COOP.CUDA_ID = "cuda:0"
    cfg.TRAINER.COOP.disable_broadcast_buffers = False
    cfg.TRAINER.COOP.bucket_cap = 25
    cfg.TRAINER.COOP.use_gradient_as_bucket_view = False
    cfg.TRAINER.COOP.Max_Batch = 0

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.xpu == 1:
        print(1)
        cfg.TRAINER.COOP.XPU = True
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # print(cfg)
    # cfg.freeze()

    return cfg


def objective(trial, cfg):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    bs = trial.suggest_int("bs", 32, 256, log=True)

    cfg.OPTIM.LR = lr
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = bs
    # cfg.freeze()
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)
    trainer.train()
    return trainer.test_acc, trainer.test_time_epoch


if __name__ == "__main__":
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument("--xpu", default=None, type=int, help="XPU id to use.")
    args = parser.parse_args()
    if args.xpu == 1:
        pass
    cfg = setup_cfg(args)
    if os.path.exists(cfg.OUTPUT_DIR):
        shutil.rmtree(cfg.OUTPUT_DIR)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    search_space = {
        "lr": [2e-3, 1e-3, 2e-2],
        "bs": [32],
    }
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(lambda trial: objective(trial, cfg), n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
    best_acc = trial_with_highest_accuracy.values[0]
    best_time = trial_with_highest_accuracy.values[1]
    best_param = trial_with_highest_accuracy.params

    for trail in study.trials:
        print(f"\tparams: {trail.params}")
        print(f"\tvalues: {trail.values}")
        if trail.values[0] >= trial_with_highest_accuracy.values[0] - 1 and trail.values[1] <= best_time:
            best_acc = trail.values[0]
            best_time = trail.values[1]
            best_param = trail.params

    print(f"\tbest_params: {best_param}")
    print(f"\tbest_acc: {best_acc}")
    print(f"\tbest_time: {best_time}")
    os._exit(1)
