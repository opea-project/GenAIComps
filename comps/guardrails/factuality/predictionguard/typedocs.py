# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from docarray import BaseDoc


class FactualityDoc(BaseDoc):
    reference: str
    text: str


class ScoreDoc(BaseDoc):
    score: float