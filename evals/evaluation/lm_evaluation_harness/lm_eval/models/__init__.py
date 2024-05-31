#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#

from . import huggingface

# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass

__all__ = [huggingface]
