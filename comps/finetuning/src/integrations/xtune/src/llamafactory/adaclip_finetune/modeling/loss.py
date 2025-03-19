# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEn(nn.Module):
    def __init__(
        self,
    ):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        # to avoid the grad disappear issue on A770 caused by the warning::
        # "UserWarning: aten::diag: an autograd kernel
        # was not registered to the Autograd key(s) but we are trying to backprop through it.
        # This may lead to silently incorrect behavior."
        # replace "logpt = torch.diag(logpt)" with torch.eye() mask
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        diag_logpt = logpt[mask]
        nce_loss = -diag_logpt
        sim_loss = nce_loss.mean()
        return sim_loss
