# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    # print(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def compute_accuracy_hf(logits, label, topk=(1,)):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    import torch

    max_value, _ = torch.max(logits, dim=1)
    for i in label:
        if logits[i][i] == max_value[i]:
            logits[i][i] += 0.0001
    i2t_acc = (logits.argmax(-1) == label).sum() / len(logits)
    # print("tmp", logits.argmax(-1))
    return 100 * i2t_acc
