# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    @property
    def out_features(self):
        """Output feature dimension."""
        if self.__dict__.get("_out_features") is None:
            return None
        return self._out_features
