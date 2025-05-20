# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import get_logits_processor
from ...extras.plotting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import (
        DataArguments,
        FinetuningArguments,
        GeneratingArguments,
        ModelArguments,
        ClipArguments,
        OptunaArguments,
    )

import gc

import optuna
import torch
import torch.distributed as dist
import torch.nn.parallel
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import collect_env_info, set_random_seed, setup_logger
from optuna.trial import TrialState

from ...clip_finetune.datasets import (
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
from ...clip_finetune.trainers import clip_adapter_hf, clip_bias_hf, clip_fullfinetune_hf, clip_vpt_hf, tip_adapter


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


def reset_cfg(cfg, args, training_args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    cfg.OUTPUT_DIR = training_args.output_dir

    cfg.SEED = training_args.seed

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


def setup_cfg(args, training_args, data_args, model_args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    if args.xpu == 1:
        cfg.TRAINER.COOP.XPU = True
        if torch.xpu.device_count() > 1:
            import intel_extension_for_pytorch
            import oneccl_bindings_for_pytorch

            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("ccl")
            os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
            os.environ["MASTER_PORT"] = "29500"  # your master port
            args.world_size = int(os.environ.get("WORLD_SIZE", -1))
            args.rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            # cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + str(local_rank)
        else:
            local_rank = 0
        if torch.distributed.is_initialized():
            torch.xpu.set_device(local_rank)
        args.xpu_id = local_rank
        args.xpu_id = "xpu:{}".format(args.xpu_id)
        cfg.TRAINER.COOP.XPU_ID = args.xpu_id
        print("xpu_id", cfg.TRAINER.COOP.XPU_ID)
    else:
        if torch.cuda.device_count() > 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"  # your master address
            os.environ["MASTER_PORT"] = "29500"  # your master port
            dist.init_process_group(
                backend="nccl",
            )
            local_rank = args.local_rank
            # cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + str(local_rank)
        else:
            local_rank = 0
        cfg.TRAINER.COOP.CUDA_ID = "cuda:{}".format(local_rank)
        print("cuda_id", cfg.TRAINER.COOP.CUDA_ID)
    cfg.output_dir = training_args.output_dir
    args.dataset_config_file = args.dataset_config_file + data_args.dataset[0] + ".yaml"
    args.config_file = args.config_file + model_args.model_name_or_path + ".yaml"
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args, training_args)

    if training_args.num_train_epochs != 3:
        cfg.OPTIM.MAX_EPOCH = int(training_args.num_train_epochs)
    if training_args.learning_rate != 5e-5:
        cfg.OPTIM.LR = training_args.learning_rate
    cfg.DATALOADER.BATCH_SIZE = args.clip_batch_size
    # print(cfg)

    return cfg


def run_sft_clip(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    clip_args: "ClipArguments",
    optuna_args: "OptunaArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):

    # Initialize our Trainer
    cfg = setup_cfg(clip_args, training_args, data_args, model_args)
    env_var_name = "CLIP_DEBUG"
    if env_var_name in os.environ:
        if os.path.exists(cfg.OUTPUT_DIR):
            shutil.rmtree(cfg.OUTPUT_DIR)
    cache_dir = os.path.join("./caches", cfg.DATASET.NAME)
    os.makedirs(cache_dir, exist_ok=True)
    cfg.TRAINER.TIP.CACHE_DIR = cache_dir
    cfg.TRAINER.TIP.CACHE_DIR_NEW = "./caches"
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    if os.path.exists(cfg.OUTPUT_DIR + "/trainer_log.jsonl"):
        os.remove(cfg.OUTPUT_DIR + "/trainer_log.jsonl")
    setup_logger(cfg.OUTPUT_DIR + "/trainer_log.jsonl")

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    cfg.DATASET.NUM_SHOTS = clip_args.few_shot_num
    cfg.TRAIN.PRINT_FREQ = clip_args.clip_logging_steps
    if clip_args.clip_bias_term is not None:
        cfg.BIAS.BIAS_TERMS = [item.strip() for item in clip_args.clip_bias_term.split(",")]
    if clip_args.clip_bias_exclude is not None:
        cfg.BIAS.BIAS_TERMS_EXCLUDE = [item.strip() for item in clip_args.clip_bias_exclude.split(",")]
    cfg.MODEL.ABS = clip_args.use_abs
    cfg.MODEL.ABS_GROUP = clip_args.use_abs_group
    cfg.MODEL.ABS_TOP = not clip_args.keep_min
    cfg.MODEL.ABS_KEEP = clip_args.keep_layers
    if clip_args.abs_group_name is not None:
        cfg.MODEL.ABS_GROUP_NAME = [item.strip() for item in clip_args.abs_group_name.split(",")]
    cfg.TRAINER.TIP.LOAD_CACHE = clip_args.tip_load_cache
    cfg.TRAINER.TIP.AUGMENT_EPOCH = clip_args.augment_epoch
    cfg.TRAINER.TIP.beta = clip_args.tip_beta
    cfg.TRAINER.TIP.alpha = clip_args.tip_alpha
    cfg.TRAINER.TIP.NEW = clip_args.new
    cfg.TRAINER.TIP.NEW_DATASET = clip_args.new_dataset
    cfg.TRAINER.TIP.search_best = clip_args.search_best
    cfg.optuna_cfg.n_trials = optuna_args.n_trials
    cfg.optuna_cfg.n_warmup_steps = optuna_args.n_warmup_steps
    cfg.optuna_cfg.sampler.name = optuna_args.sampler
    cfg.optuna_cfg.opt_params = [json.loads(optuna_args.opt_params)]

    if optuna_args.optuna == 1:
        max_epoch_log = cfg.OPTIM.MAX_EPOCH

        sampler = get_sampler(cfg)
        storage_name = "sqlite:///clip_optuna.db"
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
    if optuna_args.optuna == 1:
        print("use best param", best_param)

        for param_name, param_data in best_param.items():
            temp = []
            temp.append(param_name)
            temp.append(param_data)
            cfg.merge_from_list(temp)
        cfg.OPTIM.MAX_EPOCH = max_epoch_log

    cfg.freeze()
    print_args(clip_args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    # Training
    if not clip_args.no_train:
        trainer.train()
