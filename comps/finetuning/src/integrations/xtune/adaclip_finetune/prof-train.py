# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
import time

import torch
from configs.config import parse_with_config, parser
from modeling.clip_model import CLIP
from modeling.loss import CrossEn
from modeling.model import AdaCLIP
from optimization.utils import (
    setup_optimizer_and_scheduler_single_node,
    setup_optimizer_and_scheduler_single_node_bitfit,
    setup_optimizer_and_scheduler_single_node_lora,
)
from peft import LoraConfig, PeftModel, get_peft_model
from torch.profiler import ProfilerActivity, profile
from utils.basic_utils import set_seeds
from utils.flops_table import get_gflops_params
from utils.logger import LOGGER


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
    if (not cfg.freeze_clip) and (cfg.peft is not None) and ("lora_config" in cfg.peft):
        peft_config = LoraConfig(**cfg.lora_config)
        model = get_peft_model(model, peft_config)

    model.to(device)

    LOGGER.info("Setup model done!")
    return model, epoch


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


def zero_grads_(m):
    for n, p in m.named_parameters():
        # TODO
        pass


def train(cfg):

    set_seeds(cfg.seed)

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cpu")

    model, epochs = setup_model(cfg, device=device)
    # model = torch.compile(model)
    criterion = CrossEn()

    gflops_table = init_gflops_table(cfg)

    num_train_steps = 999
    if isinstance(model, PeftModel):
        # TODO: If needed in future, need to handle export and weight fusion
        model, optimizer, lr_scheduler = setup_optimizer_and_scheduler_single_node_lora(model, cfg, num_train_steps)
    elif cfg.peft:
        if cfg.peft.method == "bitfit":
            model, optimizer, lr_scheduler = setup_optimizer_and_scheduler_single_node_bitfit(
                model, cfg, num_train_steps
            )
        elif cfg.peft.method == "abs":
            pass
    else:
        model, optimizer, lr_scheduler = setup_optimizer_and_scheduler_single_node(model, cfg, num_train_steps)

    inputs_text_ids = torch.randint(0, 9999, (cfg.batch_size, 1, cfg.max_txt_len), device=device)
    inputs_clip = torch.randn(cfg.batch_size, cfg.num_frm, 3, 224, 224, device=device)
    inputs_policy = torch.randn(cfg.batch_size, cfg.num_frm, 3, 224, 224, device=device)

    if cfg.prof_type == "train":
        zero_grads_(model)
        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True, profile_memory=True, with_stack=True
        ) as prof_forward:
            sim_matrix, actions = model(inputs_text_ids, inputs_clip, inputs_policy, tau=cfg.init_tau, gather=False)
            loss1 = criterion(sim_matrix)
            loss2 = criterion(sim_matrix.T)
            loss = (loss1 + loss2) / 2

        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True, profile_memory=True, with_stack=True
        ) as prof_backward:
            loss.backward()

        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True, profile_memory=True, with_stack=True
        ) as prof_step:
            optimizer.step()

        # additional perf bench with perf_counter
        zero_grads_(model)
        t0 = time.perf_counter()
        sim_matrix, actions = model(inputs_text_ids, inputs_clip, inputs_policy, tau=cfg.init_tau, gather=False)
        loss1 = criterion(sim_matrix)
        loss2 = criterion(sim_matrix.T)
        loss = (loss1 + loss2) / 2
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        optimizer.step()
        t3 = time.perf_counter()
        t_elapsed = t3 - t0

        print("\n\n=========== forward ===========")
        print(prof_forward.key_averages().table(sort_by="cpu_time_total"))
        print("\n\n=========== backward ===========")
        print(prof_backward.key_averages().table(sort_by="cpu_time_total"))
        print("\n\n=========== step ===========")
        print(prof_step.key_averages().table(sort_by="cpu_time_total"))
        print(f"Forward Time: {t1 - t0:.3f}")
        print(f"Backward Time: {t2 - t1:.3f}")
        print(f"Step Time: {t3 - t2:.3f}")
        print(f"Total Time Elapsed: {t_elapsed:.3f}")

    elif cfg.prof_type == "inference":
        model.eval()
        with profile(
            activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True, profile_memory=True, with_stack=True
        ) as prof_infer:
            with torch.no_grad():
                sim_matrix, actions = model(inputs_text_ids, inputs_clip, inputs_policy, tau=cfg.init_tau, gather=False)

        # additional perf bench with perf_counter
        t0 = time.perf_counter()
        with torch.no_grad():
            sim_matrix, actions = model(inputs_text_ids, inputs_clip, inputs_policy, tau=cfg.init_tau, gather=False)
        t_elapsed = time.perf_counter() - t0

        print(prof_infer.key_averages().table(sort_by="cpu_time_total"))
        print(f"Total Time Elapsed: {t_elapsed:.3f}")


if __name__ == "__main__":
    parsed_args = parser.parse_args()
    parsed_args.top_k = 16
    parsed_args.resume = "pre-trained/didemo-c-32-16.pth"
    args = parse_with_config(parsed_args)
    if "prof_type" not in args:
        args.prof_type = "forward-backward"
    train(args)
