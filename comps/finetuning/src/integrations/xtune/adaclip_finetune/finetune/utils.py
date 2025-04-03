# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy
import torch
import torch.nn as nn
from matplotlib import pyplot
from sklearn.cluster import KMeans


def transform_importance_for_probs(vals, cfg):
    if cfg == "none":
        return vals
    else:
        method, *hparams = cfg.split("_")
        hparams = [float(_) for _ in hparams]
        miu = numpy.mean(vals)
        if method == "power":
            transformed = [numpy.power(val / miu, hparams[0]) for val in vals]
        elif method == "exp":
            std = numpy.std(vals)
            transformed = [numpy.exp((val - miu) / (std * hparams[0])) for val in vals]
        elif method == "sat":
            logvals = numpy.log(vals)
            logmiu = numpy.mean(logvals[logvals > -numpy.inf])
            logstd = numpy.std(logvals[logvals > -numpy.inf])
            transformed = [
                max(0, numpy.tanh((logval - logmiu) / (logstd * hparams[0]) + hparams[1])) for logval in logvals
            ]
        elif method == "sigmoid":
            logvals = numpy.log(vals)
            logmiu = numpy.mean(logvals[logvals > -numpy.inf])
            logstd = numpy.std(logvals[logvals > -numpy.inf])
            transformed = [
                (lambda x: 1 / (1 + numpy.exp(-x)))(numpy.tanh((logval - logmiu) / (logstd * hparams[0]) + hparams[1]))
                for logval in logvals
            ]
        elif method == "uniform":
            num_to_keep = int(hparams[0])
            p = 1.0 / num_to_keep
            th = sorted(vals, reverse=True)[num_to_keep - 1]
            transformed = [p if val >= th else 0 for val in vals]
        else:
            raise NotImplementedError(f"Probability normalization method: {method} is not implemented.")
        return transformed


def clip_param_to_be_kept(param_name, cfg):
    if "keep_module_keywords" not in cfg.peft.config:
        cfg.peft.config.keep_module_keywords = []
    if cfg.peft.method == "bitfit":
        return param_name.endswith(".bias") or any(kw in param_name for kw in cfg.peft.config.keep_module_keywords)
    elif cfg.peft.config.keep_module_keywords:
        return any(kw in param_name for kw in cfg.peft.config.keep_module_keywords)
    else:
        return True


def write_init_params_for_optimization(model, output_dir):
    with open(os.path.join(output_dir, "init_clip_params_for_optimization.txt"), "w") as f:
        f.write("{: <64s}\ttensor_shape\n".format("param_name"))
        for n, p in model.clip.named_parameters():
            if p.requires_grad:
                f.write("{: <64s}\t{}\n".format(n, ",".join([f"{_}" for _ in p.size()]) if p.size() else "1"))


def get_num_params(optimizer, model, clip=False):
    num_tunable = 0
    num_params = 0
    param_groups = optimizer.param_groups
    if clip:
        param_groups = param_groups[:-2]
    for group in param_groups:
        for p in group["params"]:
            numel = p.numel()
            if p.requires_grad:
                num_tunable += numel
    num_params = (
        sum([p.numel() for p in model.clip.parameters()]) if clip else sum([p.numel() for p in model.parameters()])
    )
    return num_tunable, num_params


class CosineBasedAngle(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=0)

    def forward(self, a, b):
        cos_theta = self.cosine(a.flatten(), b.flatten())
        theta = torch.acos(cos_theta.clamp_(-1, 1)) * 180 / torch.pi
        return theta


_importance_metric = {"angle": CosineBasedAngle(), "l2norm": lambda a, b: (a - b).norm()}


def l2norm_retouch(dists):
    return [(n, dist / numpy.sqrt(dim), dim) for n, dist, dim in dists]


_dim_normalization = {
    "angle": lambda _: _,  # angle_retouch is not stable
    "l2norm": l2norm_retouch,
}


def importance_nvd_from_weights_update(params, init_params, param_names, metric="l2norm", update_count=None):

    assert set(init_params) == set(params) == set(param_names)

    ret = []
    for n in param_names:
        p = params[n]
        p0 = init_params[n]
        dim = p.numel()
        if dim > 1:
            with torch.no_grad():
                dist = _importance_metric[metric](p, p0).item()
        else:
            dist = 0.0 if metric == "angle" else (p - p0).abs().item()
        if update_count is not None:
            dist = dist / max(update_count[n], 1)
        ret.append((n, dist, dim))

    return _dim_normalization[metric](ret)


def group_by_kmeans(nvd, num_groups=2):
    kmeans = KMeans(n_clusters=num_groups, random_state=0, n_init="auto").fit([(numpy.log(d), v) for _, v, d in nvd])
    grouped_vals = [[_[1] for i, _ in enumerate(nvd) if g == kmeans.labels_[i]] for g in range(num_groups)]
    return grouped_vals, kmeans.labels_[:]


def group_by_dim(nvd):
    dims_sorted = sorted(set([_[2] for _ in nvd]))
    labels_by_dim = {d: i for i, d in enumerate(dims_sorted)}
    labels = [labels_by_dim[d] for *_, d in nvd]
    grouped_vals = [[_[1] for i, _ in enumerate(nvd) if g == labels[i]] for g in range(len(dims_sorted))]
    return grouped_vals, labels


def groupwise_normalization(nvd, num_groups=2, stats_drop_mul_th=0.001):
    grouped_vals, labels = group_by_kmeans(nvd, num_groups=num_groups)
    init_miu = [numpy.median(vals) for vals in grouped_vals]
    grouped_vals = [
        [v for v in vals if v > m * stats_drop_mul_th] for m, vals in zip(init_miu, grouped_vals)
    ]  # remove zeros
    log_vals = [numpy.log(vals) for vals in grouped_vals]
    log_miu = [numpy.mean(_) for _ in log_vals]
    log_std = [numpy.std(_) for _ in log_vals]
    # print('DEBUG', log_miu, log_std)
    ret = [(n, numpy.exp((numpy.log(v) - log_miu[l]) / log_std[l]), d) for l, (n, v, d) in zip(labels, nvd)]
    """
    miu = [numpy.mean(vals) for vals in grouped_vals]
    if transform == 'mean':
        ret = [(n, v / miu[l], d) for l, (n, v, d) in zip(labels, nvd)]
    elif transform == 'median':
        med = [numpy.median(vals) for vals in grouped_vals]
        ret = [(n, v / med[l], d) for l, (n, v, d) in zip(labels, nvd)]
    else:
        log_vals = [numpy.log(vals) for vals in grouped_vals]
        log_miu = [numpy.mean(_) for _ in log_vals]
        log_std = [numpy.std(_) for _ in log_vals]
        ret = [(n, numpy.exp((numpy.log(v) - log_miu[l]) / log_std[l]), d) for l, (n, v, d) in zip(labels, nvd)]
    """
    return ret, labels


def visualize_param_groups(nvd, normalized_nvd, group_labels, output_dir, num_bins=50, step=None):

    fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("param (groups)")
    if group_labels is not None:
        # colors = 'brgc'
        for l_index in range(len(set(group_labels))):
            filtered_group = [nvd[i] for i, l in enumerate(group_labels) if l == l_index]
            ax1.semilogx([_[2] for _ in filtered_group], [_[1] for _ in filtered_group], ".", label=f"group{l_index}")
            if normalized_nvd is not None:
                filtered_group = [normalized_nvd[i] for i, l in enumerate(group_labels) if l == l_index]
                ax2.semilogx(
                    [_[2] for _ in filtered_group], [_[1] for _ in filtered_group], ".", label=f"group{l_index}"
                )
        ax3.hist([numpy.log10(_[1]) for _ in nvd if _[1] > 0], num_bins)
        if normalized_nvd is not None:
            ax4.hist([numpy.log10(_[1]) for _ in normalized_nvd if _[1] > 0], num_bins)
    else:
        ax1.semilogx([_[2] for _ in nvd], [_[1] for _ in nvd], "b.")
        if normalized_nvd is not None:
            ax2.semilogx([_[2] for _ in normalized_nvd], [_[1] for _ in nvd], "b.")

        ax3.hist([numpy.log10(_[1]) for _ in nvd if _[1] > 0], num_bins)
        if normalized_nvd is not None:
            ax4.hist([numpy.log10(_[1]) for _ in normalized_nvd if _[1] > 0], num_bins)
    for ax in [ax1, ax2]:
        ax.legend()
    output_name = "params"
    if step is not None:
        output_name += f"_{step}"
    pyplot.savefig(os.path.join(output_dir, output_name))
    pyplot.close(fig)
