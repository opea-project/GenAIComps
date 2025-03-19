# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .utils import groupwise_normalization, importance_nvd_from_weights_update, visualize_param_groups


class ImportanceBasedShrink:
    def __init__(
        self,
        model,
        total_steps,
        pre_steps_ratio=0.05,
        pre_batch_size=8,
        retain_ratio=0.05,
        normalization=True,
        num_groups=2,
        metric="l2norm",
        keep_module_keywords=[],
    ):
        self.init_clip_tunable = {
            n: p.cpu().detach()
            for n, p in model.clip.named_parameters()
            if not any(kw in n for kw in keep_module_keywords) and p.requires_grad
        }
        self.pre_steps_ratio = pre_steps_ratio
        self.pre_batch_size = pre_batch_size
        self.retain_ratio = retain_ratio
        self.normalization = normalization
        self.num_groups = num_groups
        self.metric = metric

    def slim_batch(self, batch):
        return batch

    def update_(step):
        pass


def importance_based_shrink_(model, epoch, output_dir, cfg):

    if "normalization" not in cfg.peft.config:
        cfg.peft.config.normalization = True
    if "num_groups" not in cfg.peft.config:
        cfg.peft.config.num_groups = 2

    normalization = cfg.peft.config.normalization
    num_groups = cfg.peft.config.num_groups
    method = cfg.peft.method

    if epoch == 0:
        params = {
            n: p
            for n, p in model.clip.named_parameters()
            if p.requires_grad and not any(kw in n for kw in cfg.peft.config.keep_module_keywords)
        }
        pth_path = os.path.join(output_dir, f"{method}-init-params.pth")
        with torch.no_grad():
            torch.save(params, pth_path)

    elif epoch == cfg.peft.config.num_pre_epochs:
        params = {
            n: p
            for n, p in model.clip.named_parameters()
            if p.requires_grad and not any(kw in n for kw in cfg.peft.config.keep_module_keywords)
        }
        param_names = sorted(params)
        pth_path = os.path.join(output_dir, f"{method}-evolved-params.pth")
        with torch.no_grad():
            torch.save(params, pth_path)
        pth_path = os.path.join(output_dir, f"{method}-init-params.pth")
        init_params = torch.load(pth_path, weights_only=False)  # , map_location='cpu')

        assert set(params) == set(init_params)

        nvd = importance_nvd_from_weights_update(params, init_params, param_names, cfg.peft.config.metric)
        num_tunable = sum([_[2] for _ in nvd])
        del init_params

        # calculate total num params & rank
        num_to_retain = round(num_tunable * cfg.peft.config.retain_ratio)

        # group & visualize
        if normalization:
            transformed_nvd, group_labels = groupwise_normalization(nvd, num_groups)
            sorted_nvd = sorted(transformed_nvd, reverse=True, key=lambda _: _[1])
        else:
            transformed_nvd = None
            group_labels = None
            sorted_nvd = sorted(nvd, reverse=True, key=lambda _: _[1])
        visualize_param_groups(nvd, transformed_nvd, group_labels, output_dir)

        # determine to be retained/dropped
        retained = []
        param_names_to_be_dropped = []
        num_retained = 0

        for i, (n, v, d) in enumerate(sorted_nvd):
            if num_retained < num_to_retain:
                num_retained += d
                l = 0 if group_labels is None else group_labels[i]
                retained.append((n, v, d, l))
            else:
                param_names_to_be_dropped.append(n)

        for n, p in model.clip.named_parameters():
            if n in param_names_to_be_dropped:
                p.requires_grad = False

        with open(os.path.join(output_dir, "retained_params.csv"), "w") as f:
            f.write("name,val,dim,group\n")
            retained = sorted(retained, key=lambda _: _[1], reverse=True)
            for _ in retained:
                f.write("{},{},{},{}\n".format(*_))

        with open(os.path.join(output_dir, "all_tunable_params.csv"), "w") as f:
            f.write("name,val,transformed_val,dim\n")
            nv_dict = {n: v for n, v, _ in nvd}
            for n, v, d in sorted_nvd:
                f.write("{},{},{},{}\n".format(n, nv_dict[n], v, d))
