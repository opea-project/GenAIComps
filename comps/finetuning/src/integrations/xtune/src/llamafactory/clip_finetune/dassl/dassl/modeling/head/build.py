# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dassl.utils import Registry, check_availability


HEAD_REGISTRY = Registry("HEAD")


def build_head(name, verbose=True, **kwargs):
    avai_heads = HEAD_REGISTRY.registered_names()
    check_availability(name, avai_heads)
    if verbose:
        print("Head: {}".format(name))
    return HEAD_REGISTRY.get(name)(**kwargs)
