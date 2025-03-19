# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .build import build_network, NETWORK_REGISTRY  # isort:skip

from .ddaig_fcn import fcn_3x32_gctx, fcn_3x32_gctx_stn, fcn_3x64_gctx, fcn_3x64_gctx_stn
