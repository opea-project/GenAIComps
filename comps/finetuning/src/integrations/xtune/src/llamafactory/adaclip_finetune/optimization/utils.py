# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from finetune import clip_param_to_be_kept
from optimization.bertadam import BertAdam
from torch.optim import AdamW


def setup_optimizer_and_scheduler(model, cfg, num_train_steps=-1):

    if hasattr(model, "module"):
        model = model.module

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in decay_clip_param_tp],
            "weight_decay": cfg.weight_decay,
            "lr": cfg.learning_rate * cfg.coef_lr,
        },
        {"params": [p for _, p in no_decay_clip_param_tp], "weight_decay": 0.0, "lr": cfg.learning_rate * cfg.coef_lr},
        {"params": [p for _, p in decay_noclip_param_tp], "weight_decay": cfg.weight_decay},
        {"params": [p for _, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
    ]

    if cfg.optim == "bertadam":
        t_total = -1 if cfg.no_warmup else num_train_steps
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            schedule="warmup_cosine",
            warmup=cfg.warmup_proportion,
            t_total=t_total,
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay
        )

    scheduler = None

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    return model, optimizer, scheduler


def setup_optimizer_and_scheduler_single_node(model, cfg, num_train_steps=-1):

    if hasattr(model, "module"):
        model = model.module

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # no_decay = ['bias', '.ln_', ]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in decay_clip_param_tp],
            "weight_decay": cfg.weight_decay,
            "lr": cfg.learning_rate * cfg.coef_lr,
        },
        {"params": [p for _, p in no_decay_clip_param_tp], "weight_decay": 0.0, "lr": cfg.learning_rate * cfg.coef_lr},
        {"params": [p for _, p in decay_noclip_param_tp], "weight_decay": cfg.weight_decay},
        {"params": [p for _, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
    ]

    if cfg.optim == "bertadam":
        t_total = -1 if cfg.no_warmup else num_train_steps
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            schedule="warmup_cosine",
            warmup=cfg.warmup_proportion,
            t_total=t_total,
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay
        )

    scheduler = None

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    return model, optimizer, scheduler


def setup_optimizer_and_scheduler_peft(model, cfg, num_train_steps=-1):

    if hasattr(model, "module"):
        model = model.module

    if cfg.peft.method in ("bitfit", "ssf"):

        for n, p in model.clip.named_parameters():
            if not clip_param_to_be_kept(n, cfg):
                p.requires_grad = False
            # TODO??? double check if this is necessary
            # else:
            #    p.requires_grad = True

    if cfg.peft.method in ("bitfit", "ssf"):
        param_optimizer = [
            (n, p)
            for n, p in model.named_parameters()
            if ("clip." not in n) or ("clip." in n and clip_param_to_be_kept(n, cfg))
        ]
    elif cfg.peft.method == "lora":
        param_optimizer = [
            (n, p) for n, p in model.named_parameters() if ("clip." in n and "lora" in n) or ("clip." not in n)
        ]
    else:
        param_optimizer = list(model.named_parameters())

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'rs_weight'] # TODO ??? LayerNorm cannot be found in clip param names
    no_decay = [
        "bias",
        "rs_weight",
        ".ln_",
    ]

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in decay_clip_param_tp],
            "weight_decay": cfg.weight_decay,
            "lr": cfg.learning_rate * cfg.coef_lr,
        },
        {"params": [p for _, p in no_decay_clip_param_tp], "weight_decay": 0.0, "lr": cfg.learning_rate * cfg.coef_lr},
        {"params": [p for _, p in decay_noclip_param_tp], "weight_decay": cfg.weight_decay},
        {"params": [p for _, p in no_decay_noclip_param_tp], "weight_decay": 0.0},
    ]

    if cfg.optim == "bertadam":
        t_total = -1 if cfg.no_warmup else num_train_steps
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            schedule="warmup_cosine",
            warmup=cfg.warmup_proportion,
            t_total=t_total,
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay
        )

    scheduler = None

    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank)

    return model, optimizer, scheduler
