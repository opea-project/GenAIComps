# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import os
import shutil
import sys

import optuna
import torch
import torch.distributed as dist
import torch.nn.parallel
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger
from datasets import (
    caltech101,
    dtd,
    flickr,
    flickr5k,
    imagenet,
    imagenet_a,
    imagenet_r,
    imagenet_sketch,
    imagenetv2,
    mini_imagenet,
    mscoco,
    oxford_pets,
)
from optuna.trial import TrialState
from trainers import clip_adapter_hf, clip_bias_hf, clip_fullfinetune_hf, clip_vpt_hf, tip_adapter

# custom


def objective(trial, cfg):
    # flag = 0
    # lr = trial.suggest_float("lr", 1e-10, 1, log=True)
    # bs = trial.suggest_int("bs", 1, 99999, log=True)
    # print(lr, bs)
    # opt_params: "{'OPTIM.LR': {'range':[1e-10, 1e-9], 'log': false}, 'DATALOADER.TRAIN_X.BATCH_SIZE': {'range':[1,3], 'log': false}}"

    need_tune_params = cfg.optuna_cfg.opt_params
    temp = []
    for param_name, param_data in need_tune_params[0].items():
        min_val, max_val = param_data["range"]
        log_scale = param_data["log"]
        if isinstance(min_val, int):
            suggested_value = trial.suggest_int(param_name, min_val, max_val, log=log_scale)
        elif isinstance(min_val, float):
            suggested_value = trial.suggest_float(param_name, min_val, max_val, log=log_scale)
        else:
            min_val = float(min_val)
            max_val = float(max_val)
            suggested_value = trial.suggest_float(param_name, min_val, max_val, log=log_scale)
        print(f"{param_name}: {suggested_value}")
        temp.append(param_name)
        temp.append(suggested_value)
    cfg.merge_from_list(temp)
    cfg.OPTIM.MAX_EPOCH = 10
    trainer = build_trainer(cfg)
    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during the trial: {e}")
    if os.path.exists(cfg.OUTPUT_DIR):
        shutil.rmtree(cfg.OUTPUT_DIR)

    trainer.optim = None
    trainer.sched = None
    gc.collect()
    trainer.model = trainer.model.to("cpu")
    for name, param in trainer.model.named_parameters():
        param.requires_grad = False
        del param.grad
        param.data = param.data.to("cpu")
    test_acc = trainer.test_acc
    test_time_epoch = trainer.test_time_epoch
    trainer.model = None
    del trainer.model
    del trainer.optim
    del trainer.sched
    del trainer
    gc.collect()
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(cfg.TRAINER.COOP.CUDA_ID)
    elif cfg.TRAINER.COOP.XPU:
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        torch.xpu.reset_max_memory_cached()
        torch.xpu.reset_max_memory_allocated()
        torch.xpu.empty_cache()
    if os.path.exists(cfg.OUTPUT_DIR):
        shutil.rmtree(cfg.OUTPUT_DIR)
    return test_acc, test_time_epoch


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
    cfg.TRAINER.COOP.ACC = 0  # do test after ACC epoch
    cfg.TRAINER.COOP.XPU = False
    cfg.TRAINER.COOP.XPU_ID = "xpu:0"
    cfg.TRAINER.COOP.CUDA_ID = "cuda:0"
    cfg.TRAINER.COOP.disable_broadcast_buffers = False
    cfg.TRAINER.COOP.bucket_cap = 25
    cfg.TRAINER.COOP.use_gradient_as_bucket_view = False
    cfg.TRAINER.COOP.Max_Batch = 0

    cfg.TRAINER.LFA = CN()
    cfg.TRAINER.LFA.UNSUP = 5
    cfg.TRAINER.LFA.USE = False
    cfg.TRAINER.LFA.FIVE_CROP = False
    cfg.TRAINER.LFA.COSINE_END_LR = 1e-7
    cfg.TRAINER.LFA.INTERPOLATE_FEATURES = False
    cfg.TRAINER.LFA.GAUSSIAN_NOISE = 0.035
    cfg.TRAINER.LFA.DROP_OUT = 0.05
    cfg.TRAINER.LFA.KNN = 3
    cfg.TRAINER.LFA.spectral_proj = False
    cfg.TRAINER.LFA.orthogonalize = False
    cfg.TRAINER.LFA.orth_beta = 0.01
    cfg.TRAINER.LFA.unsupervised = False
    cfg.TRAINER.LFA.beta_procrustes = None
    cfg.TRAINER.LFA.arerank_scale = 4.0
    cfg.TRAINER.LFA.METHOD = 0
    cfg.TRAINER.LFA.alpha = 0.9
    cfg.TRAINER.LFA.step = 0.05
    cfg.TRAINER.LFA.search_best = False

    cfg.TRAINER.TIP = CN()
    cfg.TRAINER.TIP.LOAD_CACHE = False
    cfg.TRAINER.TIP.CACHE_DIR = ""
    cfg.TRAINER.TIP.CACHE_DIR_NEW = ""
    cfg.TRAINER.TIP.AUGMENT_EPOCH = 10
    cfg.TRAINER.TIP.beta = 1.0
    cfg.TRAINER.TIP.alpha = 3.0
    cfg.TRAINER.TIP.NEW = False
    cfg.TRAINER.TIP.NEW_DATASET = False
    cfg.TRAINER.TIP.search_best = False
    cfg.TRAINER.TIP.search_scale = [12, 5]
    cfg.TRAINER.TIP.search_step = [200, 20]

    cfg.TRAINER.OPTUNA = CN()
    cfg.TRAINER.OPTUNA.LR = [2e-3, 1e-3, 2e-2]  # initialization suggest lr in optuna
    cfg.TRAINER.OPTUNA.BS = [32, 64, 128]  # initialization suggest bs in optuna

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.xpu == 1:
        cfg.TRAINER.COOP.XPU = True
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)
    # print("args.opts", args.opts)
    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # print(cfg)
    # cfg.freeze()

    return cfg


def get_sampler(cfg):
    sampler_name = cfg.optuna_cfg.sampler.name
    if sampler_name == "TPESampler":
        sampler = optuna.samplers.TPESampler()
    elif sampler_name == "CmaEsSampler":
        sampler = optuna.samplers.CmaEsSampler()
    elif sampler_name == "GPSampler":
        sampler = optuna.samplers.GPSampler()
    else:
        raise ValueError("Unknown sampler name in config")
    return sampler


def main(args):
    # initialization cfg
    cfg = setup_cfg(args)
    print(cfg)
    env_var_name = "CLIP_DEBUG"
    if env_var_name in os.environ:
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)

    # cache dir for Tip Adapter
    cache_dir = os.path.join("./caches", cfg.DATASET.NAME)
    os.makedirs(cache_dir, exist_ok=True)
    cfg.TRAINER.TIP.CACHE_DIR = cache_dir
    cfg.TRAINER.TIP.CACHE_DIR_NEW = "./caches"
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    # config for XPU, used in DDP
    if cfg.TRAINER.COOP.XPU:
        mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
        mpi_rank = int(os.environ.get("PMI_RANK", -1))
        if mpi_world_size > 0:
            os.environ["RANK"] = str(mpi_rank)
            os.environ["WORLD_SIZE"] = str(mpi_world_size)
        else:
            # set the default rank and world size to 0 and 1
            os.environ["RANK"] = str(os.environ.get("RANK", 0))
            os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))
        os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
        os.environ["MASTER_PORT"] = "29500"  # your master port
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))
        args.rank = int(os.environ.get("PMI_RANK", -1))
        init_method = "tcp://" + args.dist_url + ":" + args.dist_port
        if torch.xpu.device_count() > 1:
            mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
            mpi_rank = int(os.environ.get("PMI_RANK", -1))
            if mpi_world_size > 0:
                os.environ["RANK"] = str(mpi_rank)
                os.environ["WORLD_SIZE"] = str(mpi_world_size)
            else:
                # set the default rank and world size to 0 and 1
                os.environ["RANK"] = str(os.environ.get("RANK", 0))
                os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))
            os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
            os.environ["MASTER_PORT"] = "29500"  # your master port
            args.world_size = int(os.environ.get("WORLD_SIZE", -1))
            args.rank = int(os.environ.get("PMI_RANK", -1))
            init_method = "tcp://" + args.dist_url + ":" + args.dist_port
            dist.init_process_group(
                backend=args.dist_backend, init_method=init_method, world_size=args.world_size, rank=args.rank
            )
            local_rank = os.environ["MPI_LOCALRANKID"]
            # cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + str(local_rank)
        else:
            local_rank = 0
        args.xpu_id = local_rank
        args.xpu_id = "xpu:{}".format(args.xpu_id)
        cfg.TRAINER.COOP.XPU_ID = args.xpu_id
        print("xpu_id", cfg.TRAINER.COOP.XPU_ID)
    # config for CUDA, used in DDP
    else:
        if torch.cuda.device_count() > 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
            os.environ["MASTER_PORT"] = "29500"  # your master port
            init_method = "tcp://" + args.dist_url + ":" + args.dist_port
            dist.init_process_group(
                backend="nccl",
            )
            local_rank = args.local_rank
            # cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + str(local_rank)
        else:
            local_rank = 0
        cfg.TRAINER.COOP.CUDA_ID = "cuda:{}".format(local_rank)
        print("cuda_id", cfg.TRAINER.COOP.CUDA_ID)

    # config for optuna
    if args.use_optuna == 1:
        max_epoch_log = cfg.OPTIM.MAX_EPOCH

        sampler = get_sampler(cfg)
        storage_name = "sqlite:///optuna.db"
        study = optuna.create_study(
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=cfg.optuna_cfg.n_warmup_steps),
            directions=["maximize", "minimize"],
            study_name="clip_optuna",
            storage=storage_name,
            load_if_exists=True,
        )
        study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.optuna_cfg.n_trials, gc_after_trial=True)
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
        print(f"\thighest_accuracy: {best_acc}")
        print(f"\thighest_accuracy time: {best_time}")
        print(f"\thighest_accuracy param: {best_param}")
        for trail in study.trials:
            if trail.values is not None:
                if trail.values[0] >= trial_with_highest_accuracy.values[0] - 1 and trail.values[1] <= best_time:
                    best_acc = trail.values[0]
                    best_time = trail.values[1]
                    best_param = trail.params

        print(f"\tRelatively best_params: {best_param}")
        print(f"\tRelatively best_acc: {best_acc}")
        print(f"\tRelatively best_time: {best_time}")

    # config best param which is got from optuna
    if args.use_optuna == 1:
        print("use best param", best_param)

        for param_name, param_data in best_param.items():
            temp = []
            temp.append(param_name)
            temp.append(param_data)
            cfg.merge_from_list(temp)
        cfg.OPTIM.MAX_EPOCH = max_epoch_log

    print_args(args, cfg)
    print("Collecting env info ...")
    print("00000000000")
    print("** System info **\n{}\n".format(collect_env_info()))

    cfg.freeze()
    setup_logger(cfg.OUTPUT_DIR)
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()
    if args.use_optuna == 1:
        print("=====================finish with best param=============")
        print("finish with param lr:", best_param["lr"])
        print("finish with param bs:", best_param["bs"])
    else:
        print("=====================finish=============")


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

    parser.add_argument("--xpu", default=None, type=int, help="Whether to use XPU.")
    parser.add_argument("--xpu_id", default=None, type=int, help="XPU id to use.")
    parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--local-rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="127.0.0.1", type=str, help="url used to set up distributed training")
    parser.add_argument("--dist-port", default="29500", type=str, help="url port used to set up distributed training")
    parser.add_argument("--dist-backend", default="ccl", type=str, help="distributed backend, default is torch-ccl")
    parser.add_argument("--disable-broadcast-buffers", action="store_true", help="disable syncing buffers")
    parser.add_argument("--bucket-cap", default=25, type=int, help="controls the bucket size in MegaBytes")
    parser.add_argument(
        "--large-first-bucket",
        action="store_true",
        help="Configure a large capacity of the first bucket in DDP for allreduce",
    )
    parser.add_argument(
        "--use-gradient-as-bucket-view", action="store_true", help="Turn ON gradient_as_bucket_view optimization in DDP"
    )
    parser.add_argument("--use-optuna", default=None, type=int, help="Use optuna to get best param")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    # DDP env setup
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    print("mpi_world_size", mpi_world_size)
    if mpi_world_size > 0:
        os.environ["MASTER_ADDR"] = args.dist_url  #'127.0.0.1'
        os.environ["MASTER_PORT"] = args.dist_port  #'29500'
        os.environ["RANK"] = os.environ.get("PMI_RANK", -1)
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", -1)
        args.rank = int(os.environ.get("PMI_RANK", -1))
    args.world_size = int(os.environ.get("WORLD_SIZE", -1))
    args.distributed = args.world_size > 1
    ngpus_per_node = 1
    args.world_size = ngpus_per_node * args.world_size
    print("world_size", args.world_size)
    # Need by A770
    if args.xpu == 1:

        if args.dist_backend == "ccl":
            try:
                import oneccl_bindings_for_pytorch
            except ImportError:
                pass
    main(args)
    os._exit(1)
