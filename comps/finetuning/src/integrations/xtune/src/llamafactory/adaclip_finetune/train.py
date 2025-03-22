# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os
import sys
from datetime import datetime

import numpy as np
import optuna
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(os.path.dirname(__file__))
from modeling.clip_model import CLIP
from modeling.loss import CrossEn
from modeling.model import AdaCLIP
from torch.utils.tensorboard import SummaryWriter
from utils.basic_utils import NoOp, save_json, set_seeds
from utils.distributed import all_gather, is_main_process, reduce_loss_dict
from utils.flops_table import get_gflops_params
from utils.logger import LOGGER, add_log_to_file
from utils.train_utils import log_metrics, progress, save_checkpoint, verbose


try:
    from datasets.dataset import BaseDataset
    from datasets.prefetch import PrefetchLoader
except ImportError:
    from .datasets.dataset import BaseDataset
    from .datasets.prefetch import PrefetchLoader

# from peft import get_peft_model, LoraConfig, PeftModel
from functools import partial

from configs.config import parse_with_config, parser
from finetune import (
    LisaDispatcherForCLIPSimplified,
    get_num_params,
    groupwise_normalization,
    importance_based_shrink_,
    visualize_param_groups,
    write_init_params_for_optimization,
)
from modeling.metrics import t2v_metrics, v2t_metrics
from optimization.utils import setup_optimizer_and_scheduler_peft, setup_optimizer_and_scheduler_single_node


# for optuna sampler selection
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


# for optuna parameter tuning
def objective(trial, cfg):
    need_tune_params = cfg.optuna_cfg.opt_params
    for param_name, param_data in need_tune_params.items():
        min_val, max_val = param_data.range
        log_scale = param_data.log
        if isinstance(min_val, int):
            suggested_value = trial.suggest_int(param_name, min_val, max_val, log=log_scale)
        elif isinstance(min_val, float):
            suggested_value = trial.suggest_float(param_name, min_val, max_val, log=log_scale)
        setattr(cfg, param_name, suggested_value)
    val_best_recall = train(cfg, trial)

    return val_best_recall


def setup_model(cfg, device):
    LOGGER.info("Setup model...")

    pretrained_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    state_dict = {}
    epoch = 0
    if cfg.resume:
        LOGGER.info(f"Loading model checkpoint: {cfg.resume}...")
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint["epoch"]
    else:
        LOGGER.info("Using CLIP pretrained weights...")
        for key, val in pretrained_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        if cfg.sim_header != "meanP":
            for key, val in pretrained_state_dict.items():
                # initialize for the frame and type position embedding
                if key == "positional_embedding":
                    state_dict["frame_position_embeddings.weight"] = val.clone()

                # using weight of first 4 layers for initialization
                if key.find("transformer.resblocks") == 0:
                    num_layer = int(key.split(".")[2])

                    # initialize the 4-layer temporal transformer
                    if num_layer < 4:
                        state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                        continue

                    if num_layer == 4:  # for 1-layer transformer sim_header
                        state_dict[key.replace(str(num_layer), "0")] = val.clone()

    model = AdaCLIP(cfg, pretrained_state_dict)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix="")

    if cfg.debug:
        LOGGER.info("-" * 20)
        if len(missing_keys) > 0:
            LOGGER.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)
                )
            )
        if len(unexpected_keys) > 0:
            LOGGER.info(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)
                )
            )
        if len(error_msgs) > 0:
            LOGGER.error(
                "Weights from pretrained model cause errors in {}: {}".format(
                    model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)
                )
            )

    if str(device) == "cpu":
        model.float()

    if cfg.freeze_clip:
        model.freeze_clip()
    if cfg.freeze_cnn and cfg.use_policy:
        model.sampler.freeze_cnn_backbone()
    if (not cfg.freeze_clip) and ("peft" in cfg) and ("lora_config" in cfg.peft):
        peft_config = LoraConfig(**cfg.lora_config)
        model = get_peft_model(model, peft_config)

    model.to(device)

    LOGGER.info("Setup model done!")
    return model, epoch


def setup_dataloaders(cfg, device, train_annot, val_annot):

    LOGGER.info("Init. train_loader and val_loader...")

    train_dataset = BaseDataset(cfg, train_annot, is_train=True)
    val_dataset = BaseDataset(cfg, val_annot, is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=train_dataset.collate_data,
        pin_memory=cfg.pin_mem,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=val_dataset.collate_data,
        pin_memory=cfg.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    if str(device) != "cpu":
        train_loader = PrefetchLoader(train_loader, device)
        val_loader = PrefetchLoader(val_loader, device)

    LOGGER.info("Init. train_loader and val_loader done!")

    return train_loader, val_loader


def init_gflops_table(cfg):
    gflops_table = {}
    gflops_table["clip"] = get_gflops_params("CLIP", cfg)
    gflops_table["policy"] = get_gflops_params(cfg.policy_backbone, cfg)
    gflops_table[f"{cfg.rnn}"] = get_gflops_params(cfg.rnn, cfg) if cfg.use_rnn else 0
    gflops_table["mlp"] = get_gflops_params("mlp", cfg)

    LOGGER.info("gflops_table: ")
    for k in gflops_table:
        if k == "clip":
            LOGGER.info("%-20s: %.4f GFLOPS" % (f"{k}/f", gflops_table[k]))
        else:
            LOGGER.info("%-20s: %.4f GFLOPS" % (f"{k}/v", gflops_table[k]))

    return gflops_table


def log_policy_usage(actions, gflops_table, cfg, output_frame_index=False):
    keep_cnt = np.sum(actions == 1)
    total_cnt = actions.shape[0] * actions.shape[1]
    skip_cnt = total_cnt - keep_cnt
    LOGGER.info(f"CLIP model: {keep_cnt} ({100 * keep_cnt / total_cnt:.2f})%")
    LOGGER.info(f"Skip 1 frame: {skip_cnt} ({100 * skip_cnt / total_cnt:.2f})%")

    num_frm = cfg.num_frm if cfg.num_frm_subset <= 0 else min(cfg.num_frm, cfg.num_frm_subset)
    avg_frame_ratio = keep_cnt / total_cnt
    avg_gflops = avg_frame_ratio * gflops_table["clip"] * num_frm

    if cfg.use_policy:
        avg_gflops += gflops_table["policy"] + gflops_table[cfg.rnn] + gflops_table["mlp"]

    if cfg.use_policy and output_frame_index:
        frame_index_count = actions.sum(axis=0, dtype=np.int32)  # Obtain frame index for kept frames
        LOGGER.info(f"Out of {actions.shape[0]} videos, the number of times each frame index is being selected: ")
        LOGGER.info(frame_index_count.tolist())

    LOGGER.info(
        f"GFLOPS/f: {avg_gflops / num_frm:.3f} GFLOPS/v: {avg_gflops:.3f} AVG_FRAMES: {avg_frame_ratio * num_frm:.3f}"
    )


def get_current_temperature(cfg, epoch=0):
    return max(cfg.init_tau * np.exp(-cfg.exp_decay_factor * epoch), cfg.min_tau)


def get_current_k(cfg, k, epoch=0, warmup=0):
    if epoch >= warmup:
        return cfg.top_k
    if epoch == 0:
        return cfg.num_frm - 1
    k -= (cfg.num_frm - 1 - cfg.top_k) / warmup
    return max(cfg.top_k, k)


def get_embeddings(val_loader, model, cfg):
    with torch.no_grad():
        text_embd = []
        frame_embd = []
        word_embd = []
        actions = []
        lengths = []
        break_pts = [0]
        if is_main_process():
            pbar = tqdm(total=len(val_loader), desc="Evaluation", unit="batch")
        else:
            pbar = NoOp()

        for minibatch in val_loader:
            output = model(
                minibatch["text_input_ids"], minibatch["clip_inputs"], minibatch["policy_inputs"], return_embds=True
            )
            text_embd.append(output["text_embd"])
            frame_embd.append(output["frame_embd"])
            word_embd.append(output["word_embd"])
            actions.append(output["actions"])
            lengths.append(output["lengths"])
            pbar.update(1)
        pbar.close()

        text_embd = torch.cat(text_embd, 0)
        frame_embd = torch.cat(frame_embd, 0)
        word_embd = torch.cat(word_embd, 0) if word_embd[0] is not None else None
        actions = torch.cat(actions, 0)
        lengths = torch.cat(lengths, 0)

        if break_pts == [0]:
            break_pts = None

        res = {
            "text_embd": text_embd,
            "frame_embd": frame_embd,
            "word_embd": word_embd,
            "actions": actions,
            "lengths": lengths,
        }

        return res, break_pts


def reshape_sim_matrix(sims, break_pts):
    num_t, num_v = sims.shape
    if num_t == num_v:
        return sims
    sims_reshaped = torch.zeros((num_v, num_v)).to(sims.device)
    for v in range(num_v):
        for i in range(len(break_pts) - 1):
            sims_reshaped[i, v] = torch.max(sims[break_pts[i] : break_pts[i + 1], v], dim=0)[0]
    return sims_reshaped


def compute_batched_sim_matrix(batch_size, model, text_embd, frame_embd, word_embd, lengths, runtime=False):
    sim_matrix = []
    text_batch_size = 1 if runtime else batch_size
    video_batch_size = frame_embd.shape[0] if runtime else batch_size
    with torch.no_grad():
        for ti in range(0, text_embd.shape[0], text_batch_size):
            tf = ti + text_batch_size
            text_embd_batch = text_embd[ti:tf]
            word_embd_batch = word_embd[ti:tf] if word_embd is not None else None
            lengths_batch = lengths[ti:tf]
            each_row = []
            for vi in range(0, frame_embd.shape[0], video_batch_size):
                vf = vi + video_batch_size
                frame_embd_batch = frame_embd[vi:vf]
                sims = model.compute_sim_matrix(frame_embd_batch, text_embd_batch, word_embd_batch, lengths_batch)
                each_row.append(sims)
            each_row = torch.concat(each_row, dim=-1)
            sim_matrix.append(each_row)
        sim_matrix = torch.concat(sim_matrix, dim=0)
    return sim_matrix


@torch.no_grad()
def validate(model, val_loader, device, cfg, criterion=None, writer=None, epoch=None, gflops_table=None):
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    if cfg.use_policy and cfg.warmup_epochs:
        model.sampler.top_k = cfg.top_k

    embds, break_pts = get_embeddings(val_loader, model, cfg)

    text_embd = embds["text_embd"]
    frame_embd = embds["frame_embd"]
    word_embd = embds["word_embd"]
    actions = embds["actions"]
    lengths = embds["lengths"]

    sims = compute_batched_sim_matrix(cfg.val_batch_size, model, text_embd, frame_embd, word_embd, lengths)
    LOGGER.info(f"Num. of queries: {sims.shape[0]}, Num. of videos: {sims.shape[1]}")

    tv_metrics = t2v_metrics(sims, break_pts)
    vt_metrics = v2t_metrics(sims, break_pts)
    all_metrics = {"t2v_metrics": tv_metrics, "v2t_metrics": vt_metrics}

    if is_main_process() and criterion:
        reshaped_sims = reshape_sim_matrix(sims, break_pts)
        loss1 = criterion(reshaped_sims)
        loss2 = criterion(reshaped_sims.T)
        retrieval_loss = (loss1 + loss2) / 2
        writer.add_scalar("Retrieval Loss/val", retrieval_loss.item(), epoch)
        loss = retrieval_loss
        writer.add_scalar("Total Epoch Loss/val", loss.item(), epoch)
        LOGGER.info(f"EVAL epoch {epoch} Loss: {(loss.item()):.6f}")
        LOGGER.info(f"Retrieval Loss: {retrieval_loss.item():.3f}")
    actions = actions.cpu().detach().numpy()
    log_policy_usage(actions, gflops_table, cfg, True)

    return all_metrics, actions


def train(cfg, trial=None):

    set_seeds(cfg.seed)

    if not cfg.train_annot or not cfg.val_annot:
        raise ValueError("Empty annotation path!")

    torch.autograd.set_detect_anomaly(True)
    if cfg.xpu:

        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
    expname = os.path.split(os.path.splitext(cfg.config)[0])[-1] + f"-lr-{cfg.learning_rate * cfg.coef_lr:.2g}".replace(
        "e-0", "n"
    )
    if cfg.freeze_clip:
        expname += "-fclip"
    if cfg.optim == "adamw":
        expname += "-adamw"
    if cfg.resume:
        expname = os.path.split(cfg.resume)[-1].split("-")[0] + "-" + expname

    # expname += f"-e{cfg.num_epochs}"

    if is_main_process():
        writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, expname, timestamp))
    else:
        LOGGER.disabled = True
        writer = NoOp()

    if not cfg.no_output:
        output_dir = os.path.join(cfg.output_dir, expname, timestamp)
        add_log_to_file(os.path.join(output_dir, "log.info"))
        save_json(cfg, os.path.join(output_dir, "config.json"), save_pretty=True)

    model, epoch = setup_model(cfg, device=device)

    gflops_table = init_gflops_table(cfg)

    if cfg.do_inference:
        _, eval_loader = setup_dataloaders(cfg, device, cfg.train_annot, cfg.test_annot)
        LOGGER.info("***** Test information *****")
        LOGGER.info("  Num examples = %d", len(eval_loader.dataset))
        LOGGER.info("  Batch size = %d", cfg.batch_size)
        LOGGER.info("  Num steps = %d", len(eval_loader))

        if is_main_process():
            ret_metrics, _ = validate(model, eval_loader, device, cfg, gflops_table=gflops_table)
            for metric in ret_metrics:
                verbose(ret_metrics[metric], metric, epoch, name="TEST")

            if not cfg.no_output:
                print(f"Log file stored at {output_dir}")
        return

    train_loader, val_loader = setup_dataloaders(cfg, device, cfg.train_annot, cfg.val_annot)

    total_batch_size = int(cfg.world_size * cfg.batch_size * cfg.gradient_accumulation_steps)
    num_train_steps = int(math.ceil(1.0 * cfg.num_epochs * len(train_loader.dataset) / total_batch_size))

    LOGGER.info(f"device: {device} n_gpu: {cfg.world_size}, " f"rank: {cfg.rank}")
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training on {cfg.world_size} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_loader.dataset))
    LOGGER.info("  Batch size = %d", cfg.batch_size)
    LOGGER.info("  Accumulate steps = %d", cfg.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", num_train_steps)
    LOGGER.info("***** Validation information *****")
    LOGGER.info("  Num examples = %d", len(val_loader.dataset))
    LOGGER.info("  Batch size = %d", cfg.batch_size)
    LOGGER.info("  Num steps = %d", len(val_loader))

    assert cfg.freeze_layer_num <= 12 and cfg.freeze_layer_num >= -1
    if cfg.freeze_layer_num > -1 and not cfg.freeze_clip:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if (
                name.find("ln_final.") == 0
                or name.find("text_projection") == 0
                or name.find("logit_scale") == 0
                or name.find("visual.ln_post.") == 0
                or name.find("visual.proj") == 0
            ):
                continue  # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= cfg.freeze_layer_num:
                    continue  # need to train
            # parameters which < freeze_layer_num will be freezed
            param.requires_grad = False

    criterion = CrossEn().to(device)

    if "peft" in cfg:
        model, optimizer, lr_scheduler = setup_optimizer_and_scheduler_peft(model, cfg, num_train_steps)
    else:
        model, optimizer, lr_scheduler = setup_optimizer_and_scheduler_single_node(model, cfg, num_train_steps)

    best = -np.inf
    best_metrics, best_actions = None, None
    global_step = 0
    len_epoch = len(train_loader)
    curr_k = None

    # importance based weight
    importance_based_shrink_enabled = "peft" in cfg and cfg.peft.method == "ibs"
    lisa_enabled = "peft" in cfg and cfg.peft.method == "lisa"
    if lisa_enabled:
        lisa_dispatcher = LisaDispatcherForCLIPSimplified(model, num_train_steps, **cfg.peft.config)
        with open(os.path.join(output_dir, "lisa_tunable_clip_params.txt"), "w") as f:
            for n in lisa_dispatcher.get_lisa_tunable():
                f.write(f"{n}\n")
        os.makedirs(os.path.join(output_dir, "debug"))

    write_init_params_for_optimization(model, output_dir)

    for epoch in range(cfg.num_epochs):

        if importance_based_shrink_enabled:
            importance_based_shrink_(model, epoch, output_dir, cfg)

        set_seeds(cfg.seed + epoch)

        total_loss = 0
        model.train()

        # train_loader.sampler.set_epoch(epoch)
        all_actions_list = []
        tau = get_current_temperature(cfg, epoch)

        if cfg.use_policy and cfg.warmup_epochs:
            curr_k = get_current_k(cfg, curr_k, epoch, cfg.warmup_epochs)
            if hasattr(model, "module"):
                model.module.sampler.top_k = curr_k
            else:
                model.sampler.top_k = curr_k

        for step, minibatch in enumerate(train_loader):
            if importance_based_shrink_enabled and epoch < cfg.peft.config.num_pre_epochs:
                sub_indices = torch.randperm(minibatch["clip_inputs"].size(0))[: cfg.peft.config.pre_batch_size]
                minibatch = {k: v[sub_indices] for k, v in minibatch.items()}

            sim_matrix, actions = model(
                minibatch["text_input_ids"], minibatch["clip_inputs"], minibatch["policy_inputs"], tau=tau, gather=False
            )
            all_actions_list.append(actions.cpu().detach().numpy())
            loss1 = criterion(sim_matrix)
            loss2 = criterion(sim_matrix.T)
            retrieval_loss = (loss1 + loss2) / 2

            loss = retrieval_loss

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            if lisa_enabled:
                if lisa_dispatcher.update_(model, global_step):
                    num_tunable, num_total = get_num_params(optimizer, model, clip=True)
                    writer.add_scalar("CLIP tunable ratio", num_tunable / num_total, global_step)
                if global_step % cfg.peft.config.log_probs_interval == 0:
                    lisa_dispatcher.write_probs(model, os.path.join(output_dir, "debug", f"{global_step:0>6d}.csv"))
                    gn_nvd, labels = groupwise_normalization(lisa_dispatcher.importance_nvd, lisa_dispatcher.num_groups)
                    visualize_param_groups(
                        lisa_dispatcher.importance_nvd,
                        gn_nvd,
                        labels,
                        os.path.join(output_dir, "debug"),
                        step=global_step,
                    )

            # Reduce losses over all GPUs for logging purposes
            loss_dict = {"Retrieval Loss": retrieval_loss}
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = loss_dict_reduced["Retrieval Loss"]
            total_loss += losses_reduced.item()

            global_step += 1

            if (step + 1) % cfg.gradient_accumulation_steps == 0:

                if cfg.grad_norm != -1:
                    clip_grad_norm_(model.parameters(), cfg.grad_norm)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                optimizer.step()
                optimizer.zero_grad()

                # https://github.com/openai/CLIP/issues/46
                if hasattr(model, "module"):
                    torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
                else:
                    torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

                if cfg.optim == "bertadam":
                    lrs = optimizer.get_lrs()
                if global_step % cfg.n_display == 0:
                    prog = progress(step + 1, len_epoch)
                    lr = (
                        "|".join([f"{itm:.2e}" for itm in sorted(lrs)])
                        if cfg.optim == "bertadam"
                        else cfg.learning_rate
                    )
                    LOGGER.info(f"Train Epoch: {epoch} {prog} Loss: {losses_reduced.item():.6f} Lr: {lr}")
                    LOGGER.info("  ".join([f"{k}: {v.item():.3f}" for k, v in loss_dict_reduced.items()]))
                    if cfg.use_policy:
                        LOGGER.info(f"Gumbel softmax temperature: {tau:.4f}")
                    log_policy_usage(actions.cpu().detach().numpy(), gflops_table, cfg)
                writer.add_scalar("Retrieval Loss/train", loss_dict_reduced["Retrieval Loss"].item(), global_step)
                writer.add_scalar("Total Loss/train", losses_reduced.item(), global_step)
                # writer.add_scalar('LR/max', max(lrs), global_step)
                # writer.add_scalar('LR/min', min(lrs), global_step)

        LOGGER.info(f"Train Epoch: {epoch} Loss: {(total_loss / len_epoch):.6f}")
        num_tunable, num_total = get_num_params(optimizer, model, clip=False)
        LOGGER.info(
            "# number of tunable/total params: {}/{}={:.3f}%".format(
                num_tunable, num_total, num_tunable / num_total * 100.0
            )
        )
        num_tunable, num_total = get_num_params(optimizer, model, clip=True)
        LOGGER.info(
            "# number of tunable/total CLIP params: {}/{}={:.3f}%".format(
                num_tunable, num_total, num_tunable / num_total * 100.0
            )
        )
        writer.add_scalar("Total Epoch Loss/train", total_loss / len_epoch, epoch)
        log_policy_usage(np.concatenate(all_actions_list, axis=0), gflops_table, cfg, True)

        set_seeds(cfg.seed)
        if is_main_process():
            ret_metrics, val_actions = validate(model, val_loader, device, cfg, criterion, writer, epoch, gflops_table)
            for metric in ret_metrics:
                verbose(ret_metrics[metric], metric, epoch)
                log_metrics(ret_metrics[metric], metric, epoch, writer)

            best_recall = (ret_metrics["t2v_metrics"]["R1"] + ret_metrics["t2v_metrics"]["R5"]) / 2
            if trial is not None:
                # record each epoch's best_recall
                trial.report(best_recall, epoch)
                # handle pruning based on intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            improved = best_recall > best

            if improved:
                best = best_recall
                best_metrics = ret_metrics
                best_actions = val_actions
                best_checkpoint = {"epoch": epoch, "model": model}
                if not cfg.no_output:
                    save_checkpoint(best_checkpoint, cfg, optimizer, os.path.join(output_dir, "trained_model.pth"))
                    LOGGER.info(f"Saving the best ckpt to disk (epoch {best_checkpoint['epoch']})")
            else:
                LOGGER.info(
                    f"This epoch did not improve R1-5-10. Best checkpoint saved for epoch {best_checkpoint['epoch']}"
                )

            if cfg.save_last and epoch == cfg.num_epochs - 1:
                last_checkpoint = {"epoch": epoch, "model": model}
                save_checkpoint(last_checkpoint, cfg, optimizer, os.path.join(output_dir, "trained_model_last.pth"))

    if is_main_process():
        writer.close()
        LOGGER.info(f"Best retrieval performance from epoch {best_checkpoint['epoch']}")
        log_policy_usage(best_actions, gflops_table, cfg, True)
        for metric in best_metrics:
            verbose(best_metrics[metric], metric, best_checkpoint["epoch"])

    if not cfg.no_output:
        output_dirs = all_gather(output_dir)
        if is_main_process():
            if cfg.test_annot != cfg.val_annot:
                _, eval_loader = setup_dataloaders(cfg, device, cfg.train_annot, cfg.test_annot)
                LOGGER.info("***** Test information *****")
                LOGGER.info("  Num examples = %d", len(eval_loader.dataset))
                LOGGER.info("  Batch size = %d", cfg.batch_size)
                LOGGER.info("  Num steps = %d", len(eval_loader))

                if is_main_process():
                    set_seeds(cfg.seed)
                    cfg.resume = os.path.join(output_dirs[0], "trained_model.pth")
                    model, epoch = setup_model(cfg, device)
                    ret_metrics, _ = validate(model, eval_loader, device, cfg, gflops_table=gflops_table)
                    for metric in ret_metrics:
                        verbose(ret_metrics[metric], metric, epoch, name="TEST")

            LOGGER.info(f"Log file and the best performing ckpt can be found at {str(output_dirs[0])}")

    return best


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    args = parse_with_config(parsed_args)
    if "optuna_cfg" in args:
        args.optuna = True
    if args.optuna:
        # do optuna to tune parameters in the config file
        storage_name = "sqlite:///optuna.db"
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
