# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from ..build import DATASET_REGISTRY
from .cifar import CIFAR10


@DATASET_REGISTRY.register()
class SVHN(CIFAR10):
    """SVHN for SSL.

    Reference:
        - Netzer et al. Reading Digits in Natural Images with
        Unsupervised Feature Learning. NIPS-W 2011.
    """

    dataset_dir = "svhn"

    def __init__(self, cfg):
        super().__init__(cfg)
