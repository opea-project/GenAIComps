#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#

from .accuracy import evaluate
from .arguments import BigcodeEvalParser, setup_parser

__all__ = [evaluate, BigcodeEvalParser, setup_parser]
