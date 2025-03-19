# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import copy
import errno
import json
import os
import os.path as osp
import random
import sys
import time
import warnings
from difflib import SequenceMatcher

import numpy as np
import PIL
import torch
from PIL import Image


try:
    from geomloss import SamplesLoss
except ImportError:
    print("Could not import geomloss, U-LFA cannot be run")

__all__ = [
    "mkdir_if_missing",
    "check_isfile",
    "read_json",
    "write_json",
    "set_random_seed",
    "download_url",
    "read_image",
    "collect_env_info",
    "listdir_nohidden",
    "get_most_similar_str_to_a_from_b",
    "check_availability",
    "tolist_if_not",
    "sinkhorn_assignment",
    "l2_norm",
    "get_accuracies",
    "get_one_to_one_features",
]


def get_one_to_one_features(visual_features, class_prototypes, labels):
    if labels is not None:
        text_features = class_prototypes[labels]
        return text_features

    assignments = sinkhorn_assignment(visual_features, class_prototypes)
    text_features = assignments @ class_prototypes
    return text_features


def topk_accuracies(preds, labels, topk_vals):
    assert preds.size(0) == labels.size(0)

    # Get class index for the top k probabilities
    top_max_k_inds = torch.topk(preds, max(topk_vals), dim=1, largest=True, sorted=True)[1]

    top_max_k_inds = top_max_k_inds.t()
    # duplicate the prediction top k time
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)

    # count the number of correct predictions
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    num_topks_correct = [top_max_k_correct[:k, :].float().sum() for k in topk_vals]

    accuracies = [(x / preds.size(0)) * 100.0 for x in num_topks_correct]
    return accuracies


def get_acc(visual_feats, class_prototypes, labels, alpha=0.0, topk=(1, 5), softmax_temp=0.07):
    assert visual_feats.ndim == class_prototypes.ndim == 2
    assert visual_feats.size(1) == class_prototypes.size(1)

    logits = (visual_feats @ class_prototypes.T) / softmax_temp
    probabilities = logits.softmax(-1)
    topk_accs = topk_accuracies(probabilities, labels, topk)
    topk_accs = [acc.cpu().numpy().round(2) for acc in topk_accs]
    return {f"top_{i}": acc for i, acc in zip(topk, topk_accs)}


def get_acc_new(
    cfg, new_visual_feats, ori_visual_feats, class_prototypes, labels, alpha=0.0, topk=(1, 5), softmax_temp=0.07
):
    assert new_visual_feats.ndim == class_prototypes.ndim == 2
    assert new_visual_feats.size(1) == class_prototypes.size(1)

    ori_logits = (ori_visual_feats @ class_prototypes.T) / softmax_temp
    new_logits = (new_visual_feats @ class_prototypes.T) / softmax_temp
    logits = (1 - alpha) * ori_logits + alpha * new_logits
    probabilities = logits.softmax(-1)
    topk_accs = topk_accuracies(probabilities, labels, topk)
    topk_accs = [acc.cpu().numpy().round(2) for acc in topk_accs]
    return {f"top_{i}": acc for i, acc in zip(topk, topk_accs)}


def get_accuracies(
    cfg, train_arrays, test_arrays, transform=None, target_set_transform=None, five_crop=False, alpha=0.0
):
    device = cfg.TRAINER.COOP.XPU_ID if cfg.TRAINER.COOP.XPU else cfg.TRAINER.COOP.CUDA_ID

    names = ["train", "test"]

    arrays = [train_arrays] + [test_arrays]
    accuracies = {}
    for arr, name in zip(arrays, names):

        class_prototypes = arr["text_features"].to(device)
        visual_feats = arr["visual_features"].to(device)
        labels = arr["labels"].to(device)
        if five_crop and name == "train":
            # only keep the center ones for fast testing
            mask = torch.tensor([0, 0, 0, 0, 1]).repeat(visual_feats.shape[0] // 5).bool()
            mask = torch.cat([mask, torch.zeros(visual_feats.shape[0] - mask.shape[0])], dim=0).bool()
            visual_feats = visual_feats[mask]
            labels = labels[mask]

        class_prototypes = class_prototypes.to(device)
        visual_feats = visual_feats.to(device)
        new_visual_feats = copy.deepcopy(visual_feats)
        labels = labels.to(device)

        if target_set_transform is not None and name == "new":
            new_visual_feats = l2_norm(new_visual_feats @ target_set_transform.to(device))

        elif transform is not None:
            # renormalize since transform might not be orthogonal
            new_visual_feats = l2_norm(new_visual_feats @ transform.to(device))
        if alpha == 0.0:
            accuracies[name] = get_acc(new_visual_feats, class_prototypes, labels, alpha)
        else:
            if cfg.TRAINER.LFA.METHOD == 1:
                new_visual_feats = (1 - alpha) * visual_feats + alpha * new_visual_feats
            if cfg.TRAINER.LFA.METHOD != 2:
                accuracies[name] = get_acc(new_visual_feats, class_prototypes, labels, alpha)
            else:
                accuracies[name] = get_acc_new(cfg, new_visual_feats, visual_feats, class_prototypes, labels, alpha)

    return accuracies


def l2_norm(features):
    return features / features.norm(dim=-1, p=2, keepdim=True)


def sinkhorn_assignment(cfg, x_source, y_target, p=2, blur=0.05, scaling=0.95, batch=1000, verbose=True):
    device = cfg.TRAINER.COOP.XPU_ID if cfg.TRAINER.COOP.XPU else cfg.TRAINER.COOP.CUDA_ID
    if verbose:
        print("Generating the assignment with sinkhorn ...")
    N, M, D = x_source.shape[0], y_target.shape[0], x_source.shape[1]
    x_source = x_source.to(device)
    y_target = y_target.to(device)

    x_source_w = torch.ones(N, device=x_source.device) / N
    y_target_w = torch.ones(M, device=x_source.device) / M

    sinkhorn_solver = SamplesLoss(loss="sinkhorn", p=p, blur=blur, scaling=scaling, debias=False, potentials=True)

    F, G = sinkhorn_solver(x_source_w, x_source, y_target_w, y_target)
    x_source = x_source.view(N, 1, D)
    x_source_weights = x_source_w.view(N, 1)

    y_target = y_target.view(1, M, D)
    y_target_weights = y_target_w.view(1, M)

    F, G = F.view(N, 1), G.view(1, M)

    soft_assignments = torch.zeros(N, M, device=x_source.device)

    for i in range(0, N, batch):
        # loop to avoid memory issues

        cost_matrix = (1 / p) * ((x_source[i : i + batch] - y_target) ** p).sum(-1)  # (N,M)
        eps = blur**p  # temperature epsilon

        # (N,M) transport plan
        transport_plan = ((F[i : i + batch] + G - cost_matrix) / eps).exp()
        transport_plan = transport_plan * (x_source_weights[i : i + batch] * y_target_weights)

        soft_assignments[i : i + batch] = transport_plan / transport_plan.sum(dim=1, keepdim=True)

    return soft_assignments.cpu()


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Check if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_url(url, dst):
    """Download file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed" % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    return Image.open(path).convert("RGB")


def collect_env_info():
    """Return env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen


def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "The requested one is expected "
            "to belong to {}, but got [{}] "
            "(do you mean [{}]?)".format(available, requested, psb_ans)
        )


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x
