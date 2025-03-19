# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .backbone import BACKBONE_REGISTRY, Backbone, build_backbone
from .head import HEAD_REGISTRY, build_head
from .network import NETWORK_REGISTRY, build_network
