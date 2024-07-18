# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from pydantic import BaseModel, Field


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str
