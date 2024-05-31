#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#

from .accuracy import cli_evaluate as evaluate
from .arguments import LMEvalParser, setup_parser

__all__ = [evaluate, LMEvalParser, setup_parser]
