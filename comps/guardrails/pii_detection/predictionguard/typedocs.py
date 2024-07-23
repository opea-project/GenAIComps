# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from docarray import BaseDoc


class PIIDoc(BaseDoc):
    prompt: str
    replace: bool
    replace_method: Optional[str] = "random"