# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random

import numpy
from numpy.random import choice

from .utils import groupwise_normalization, importance_nvd_from_weights_update, transform_importance_for_probs


class LisaDispatcherForCLIPSimplified:
    def __init__(
        self,
        model,
        total_steps,
        active_ratio=0.05,
        sampling_interval=7,
        probs_update_func="sin-1",
        keep_module_keywords=[],
        normalization=True,
        num_groups=2,
        metric="l2norm",
        warmup_steps=500,
        warmup_decay=0.1,
        accu_decay=False,
        prob_transform="power-2",
        **_,
    ):
        self.init_clip_tunable = {
            n: p.cpu().detach()
            for n, p in model.clip.named_parameters()
            if not any(kw in n for kw in keep_module_keywords) and p.requires_grad
        }
        self.clip_tunable_names = sorted(self.init_clip_tunable)
        self.num_tunable = len(self.init_clip_tunable)
        self.active_ratio = active_ratio
        self.num_active = round(self.num_tunable * active_ratio)
        self.metric = metric
        self.sampling_probs = numpy.array(
            [
                1 / self.num_tunable,
            ]
            * self.num_tunable
        )
        self.importance_nvd = [(n, 0, self.init_clip_tunable[n].numel()) for n in self.clip_tunable_names]
        self.importance = [_[1] for _ in self.importance_nvd]
        self.transformed_importance = {n: 0 for n in self.clip_tunable_names}

        self.sampling_interval = sampling_interval
        self.update_count = {n: 0 for n in self.clip_tunable_names}

        self.normalization = normalization
        self.num_groups = num_groups
        self.group_labels = None
        self.prob_transform = prob_transform

        assert warmup_steps >= 0
        self.total_steps = total_steps - warmup_steps
        self.probs_update_func = self._get_probs_update_func(probs_update_func)

        self.warmup_steps = warmup_steps
        self.warmup_decay = warmup_decay
        self.warmup = True if warmup_steps > 0 else False
        self.accu_decay = accu_decay

    def _get_probs_update_func(self, func_expr):
        func_name, func_arg = func_expr.split("_")
        max_val = float(func_arg)
        if func_name == "sin":
            return lambda step: max_val * numpy.sin(
                0.5 * max(0, step - self.warmup_steps) / self.total_steps * numpy.pi
            )
        if func_name == "linear":
            return lambda step: max_val * max(0, step - self.warmup_steps) / self.total_steps
        if func_name == "square":
            return lambda step: max_val * (max(0, step - self.warmup_steps) / self.total_steps) ** 2
        if func_name == "const":
            return lambda _: max_val
        else:
            raise NotImplementedError(f"Function {func_name} is not implemented.")

    def update_(self, model, step):
        if step % self.sampling_interval == 0:
            if not self.warmup:
                self.update_sampling_probs(model, step)
            act_params = self.switch_active_layers_(model)
            for act_param in act_params:
                self.update_count[act_param] += self.sampling_interval
                if self.warmup:
                    if step < self.warmup_steps:
                        self.sampling_probs[self.clip_tunable_names.index(act_param)] *= self.warmup_decay
                    else:
                        # reset to equal
                        self.sampling_probs = numpy.ones_like(self.sampling_probs) / self.sampling_probs.size
                        self.warmup = False

            if self.warmup and step < self.warmup_steps:
                self.sampling_probs /= self.sampling_probs.sum()

            return True
        else:
            return False

    def switch_active_layers_(self, model):
        act_params = choice(self.clip_tunable_names, self.num_active, replace=False, p=self.sampling_probs)

        for n, p in model.clip.named_parameters():
            if n in self.init_clip_tunable:
                if n in act_params:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        return act_params

    def cal_sampling_probs(self, model):
        clip_tunable = {n: p.cpu().detach() for n, p in model.clip.named_parameters() if n in self.clip_tunable_names}
        self.importance_nvd = importance_nvd_from_weights_update(
            clip_tunable,
            self.init_clip_tunable,
            self.clip_tunable_names,
            metric=self.metric,
            update_count=self.update_count if self.accu_decay else None,
        )
        if self.normalization:
            gn_importance_nvd, self.group_labels = groupwise_normalization(self.importance_nvd, self.num_groups)
            self.importance = [_[1] for _ in gn_importance_nvd]
        else:
            self.importance = [_[1] for _ in self.importance_nvd]
        self.transformed_importance = transform_importance_for_probs(self.importance, self.prob_transform)
        return numpy.array(self.transformed_importance) / sum(self.transformed_importance)

    def update_sampling_probs(self, model, step):
        probs = self.cal_sampling_probs(model)
        if not (any(numpy.isnan(probs)) or any(numpy.isinf(probs))):
            probs_update_coeff = self.probs_update_func(step)
            self.sampling_probs = self.sampling_probs * (1 - probs_update_coeff) + probs * probs_update_coeff

    def get_lisa_tunable(self):
        return self.clip_tunable_names[:]

    def write_probs(self, model, filepath):
        with open(filepath, "w") as f:
            f.write("param,sampling_probs,probs,importance,transformed_importance,count\n")
            probs = self.cal_sampling_probs(model)
            for i, (n, p) in enumerate(
                sorted(zip(self.clip_tunable_names, self.sampling_probs), reverse=True, key=lambda _: _[1])
            ):
                f.write(
                    f"{n},{p:.6g},{probs[i]:.6g},{self.importance[i]:.6g},{self.transformed_importance[i]:.6g},{self.update_count[n]}\n"
                )


class LisaDispatcherForCLIPSimplifiedG:
    def __init__(
        self,
        model,
        total_steps,
        active_ratio=0.05,
        sampling_interval=7,
        probs_update_func="sin_1",
        keep_module_keywords=[],
        normalization=True,
        num_groups=2,
        metric="l2norm",
        warmup_steps=500,
        warmup_decay=0.1,
        prob_transform="power-2",
        **_,
    ):
        self.clip_tunable = {
            n: p
            for n, p in model.clip.named_parameters()
            if not any(kw in n for kw in keep_module_keywords) and p.requires_grad
        }
        self.clip_tunable_names = sorted(self.clip_tunable)
        self.num_tunable = len(self.clip_tunable)
        self.active_ratio = active_ratio
        self.num_active = round(self.num_tunable * active_ratio)
        self.metric = metric
        self.sampling_probs = numpy.array(
            [
                1 / self.num_tunable,
            ]
            * self.num_tunable
        )
        self.importance_nvd = [(n, 0, self.clip_tunable[n].numel()) for n in self.clip_tunable_names]
        self.importance = [_[1] for _ in self.importance_nvd]
        self.transformed_importance = {n: 0 for n in self.clip_tunable_names}

        self.sampling_interval = sampling_interval
        self.update_count = {n: 0 for n in self.clip_tunable_names}

        self.normalization = normalization
        self.num_groups = num_groups
        self.group_labels = None
        self.prob_transform = prob_transform

        assert warmup_steps >= 0
        self.total_steps = total_steps - warmup_steps
        self.probs_update_func = self._get_probs_update_func(probs_update_func)

        self.warmup_steps = warmup_steps
        self.warmup_decay = warmup_decay
        self.warmup = True if warmup_steps > 0 else False

    def _get_probs_update_func(self, func_expr):
        func_name, func_arg = func_expr.split("_")
        max_val = float(func_arg)
        if func_name == "sin":
            return lambda step: max_val * numpy.sin(
                0.5 * max(0, step - self.warmup_steps) / self.total_steps * numpy.pi
            )
        if func_name == "linear":
            return lambda step: max_val * max(0, step - self.warmup_steps) / self.total_steps
        if func_name == "square":
            return lambda step: max_val * (max(0, step - self.warmup_steps) / self.total_steps) ** 2
        if func_name == "const":
            return lambda _: max_val
        else:
            raise NotImplementedError(f"Function {func_name} is not implemented.")

    def update_(self, model, step):
        if step % self.sampling_interval == 0:
            if not self.warmup and step > 0:
                self.update_sampling_probs(model, step)
            act_params = self.switch_active_layers_(model)
            for act_param in act_params:
                self.update_count[act_param] += self.sampling_interval
                if self.warmup:
                    if step < self.warmup_steps:
                        self.sampling_probs[self.clip_tunable_names.index(act_param)] *= self.warmup_decay
                    else:
                        # reset to equal
                        self.sampling_probs = numpy.ones_like(self.sampling_probs) / self.sampling_probs.size
                        self.warmup = False

            if self.warmup and step < self.warmup_steps:
                self.sampling_probs /= self.sampling_probs.sum()

            return True
        else:
            return False

    def switch_active_layers_(self, model):
        act_params = choice(self.clip_tunable_names, self.num_active, replace=False, p=self.sampling_probs)

        for n, p in model.clip.named_parameters():
            if n in self.clip_tunable:
                if n in act_params:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        return act_params

    @staticmethod
    def _cal_new_importance_val(param, count, val):
        if param.requires_grad and param.grad is not None:
            c = 1.0 / max(count, 1)
            return c * param.grad.norm().item() / numpy.sqrt(param.numel()) + (1 - c) * val
        else:
            return val

    def cal_sampling_probs(self, model):
        self.importance_nvd = [
            (n, self._cal_new_importance_val(self.clip_tunable[n], self.update_count[n], v), d)
            for n, v, d in self.importance_nvd
        ]
        if self.normalization:
            gn_importance_nvd, self.group_labels = groupwise_normalization(self.importance_nvd, self.num_groups)
            self.importance = [_[1] for _ in gn_importance_nvd]
        else:
            self.importance = [_[1] for _ in self.importance_nvd]
        self.transformed_importance = transform_importance_for_probs(self.importance, self.prob_transform)
        return numpy.array(self.transformed_importance) / sum(self.transformed_importance)

    def update_sampling_probs(self, model, step):
        probs = self.cal_sampling_probs(model)
        if not (any(numpy.isnan(probs)) or any(numpy.isinf(probs))):
            probs_update_coeff = self.probs_update_func(step)
            self.sampling_probs = self.sampling_probs * (1 - probs_update_coeff) + probs * probs_update_coeff

    def get_lisa_tunable(self):
        return self.clip_tunable_names[:]

    def write_probs(self, model, filepath):
        with open(filepath, "w") as f:
            f.write("param,sampling_probs,probs,importance,transformed_importance,count\n")
            probs = self.cal_sampling_probs(model)
            for i, (n, p) in enumerate(
                sorted(zip(self.clip_tunable_names, self.sampling_probs), reverse=True, key=lambda _: _[1])
            ):
                f.write(
                    f"{n},{p:.6g},{probs[i]:.6g},{self.importance[i]:.6g},{self.transformed_importance[i]:.6g},{self.update_count[n]}\n"
                )


###################### DEPRECIATED #########################
# ⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇⬇#


class LisaDispatcherForCLIP:
    def __init__(
        self,
        model,
        active_ratio=0.05,
        sampling_interval=7,
        probs_update_coeff=0,
        probs_update_interval=-1,
        keep_module_keywords=[],
        boot_rounds=5,
        normalization=True,
        num_groups=2,
        metric="l2norm",
        prob_transform="power-2",
        **_,
    ):
        self.init_clip_tunable = {
            n: p.cpu().detach()
            for n, p in model.clip.named_parameters()
            if not any(kw in n for kw in keep_module_keywords) and p.requires_grad
        }
        self.clip_tunable_names = sorted(self.init_clip_tunable)
        self.num_tunable = len(self.init_clip_tunable)
        self.active_ratio = active_ratio
        self.num_active = round(self.num_tunable * active_ratio)
        self.probs_update_coeff = probs_update_coeff
        self.metric = metric
        self.sampling_probs = numpy.array(
            [
                1 / self.num_tunable,
            ]
            * self.num_tunable
        )
        self.importance_nvd = [(n, 0, self.init_clip_tunable[n].numel()) for n in self.clip_tunable_names]
        self.importance = [_[1] for _ in self.importance_nvd]
        self.transformed_importance = {n: 0 for n in self.clip_tunable_names}

        self.sampling_interval = sampling_interval
        self.probs_update_interval = probs_update_interval

        self.boot_sampled = 0
        self.boot_sample_index = 0
        self.total_boot_rounds = boot_rounds * self.num_tunable
        self.update_count = {n: 0 for n in self.clip_tunable_names}
        self.is_booting = True if self.probs_update_coeff > 0 else False
        self.shuffled_clip_tunable_names = self.clip_tunable_names[:]
        random.shuffle(self.shuffled_clip_tunable_names)

        self.sampling_steps = -self.probs_update_interval + 1

        self.normalization = normalization
        self.num_groups = num_groups
        self.group_labels = None
        self.prob_transform = prob_transform

    def update_(self, model, step):
        if self.probs_update_coeff > 0:
            if not self.is_booting:
                if self.sampling_steps % self.probs_update_interval == 0:
                    self.update_sampling_probs(model)
                self.sampling_steps += 1
        if step % self.sampling_interval == 0:
            act_params = self.switch_active_layers_(model)
            for act_param in act_params:
                self.update_count[act_param] += self.sampling_interval
            return True
        return False

    def uniform_sampling_without_replacement(self):
        if self.num_active > self.num_tunable - self.boot_sample_index:
            # next cycle & shuffle
            act_params = self.shuffled_clip_tunable_names[self.boot_sample_index :]
            random.shuffle(self.shuffled_clip_tunable_names)
            self.boot_sample_index = self.num_active - len(act_params)
            act_params += self.shuffled_clip_tunable_names[: self.boot_sample_index]
        else:
            act_params = self.shuffled_clip_tunable_names[
                self.boot_sample_index : self.boot_sample_index + self.num_active
            ]
            self.boot_sample_index += self.num_active

        self.boot_sampled += self.num_active

        return act_params

    def switch_active_layers_(self, model):
        if self.probs_update_coeff > 0:
            if self.is_booting:
                if self.boot_sampled < self.total_boot_rounds:
                    act_params = self.uniform_sampling_without_replacement()
                    if self.boot_sampled > self.total_boot_rounds:
                        num_active = self.num_active - (self.boot_sampled - self.total_boot_rounds)
                        act_params = act_params[:num_active]
                else:
                    self.update_sampling_probs(model)
                    act_params = choice(self.clip_tunable_names, self.num_active, replace=False, p=self.sampling_probs)
                    self.is_booting = False
            else:
                act_params = choice(self.clip_tunable_names, self.num_active, replace=False, p=self.sampling_probs)
        else:
            act_params = self.uniform_sampling_without_replacement()

        for n, p in model.clip.named_parameters():
            if n in self.init_clip_tunable:
                if n in act_params:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        return act_params

    def cal_sampling_probs(self, model):
        clip_tunable = {n: p.cpu().detach() for n, p in model.clip.named_parameters() if n in self.clip_tunable_names}
        self.importance_nvd = importance_nvd_from_weights_update(
            clip_tunable, self.init_clip_tunable, self.clip_tunable_names, metric=self.metric
        )
        if self.normalization:
            gn_importance_nvd, self.group_labels = groupwise_normalization(self.importance_nvd, self.num_groups)
            self.importance = [_[1] for _ in gn_importance_nvd]
        else:
            self.importance = [_[1] for _ in self.importance_nvd]
        self.transformed_importance = transform_importance_for_probs(self.importance, self.prob_transform)
        return numpy.array(self.transformed_importance) / sum(self.transformed_importance)

    def update_sampling_probs(self, model):
        probs = self.cal_sampling_probs(model)
        self.sampling_probs = self.sampling_probs * (1 - self.probs_update_coeff) + probs * self.probs_update_coeff

    def get_lisa_tunable(self):
        return self.clip_tunable_names[:]

    def write_probs(self, model, filepath):
        with open(filepath, "w") as f:
            f.write("param,sampling_probs,probs,importance,transformed_importance,count\n")
            probs = self.cal_sampling_probs(model)
            for i, (n, p) in enumerate(
                sorted(zip(self.clip_tunable_names, self.sampling_probs), reverse=True, key=lambda _: _[1])
            ):
                f.write(
                    f"{n},{p:.6g},{probs[i]:.6g},{self.importance[i]:.6g},{self.transformed_importance[i]:.6g},{self.update_count[n]}\n"
                )
