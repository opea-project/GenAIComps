# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from opea_haystack.utils import OPEABackend


class MockBackend(OPEABackend):
    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__(api_url="", model_kwargs=model_kwargs or {})

    def embed(self, texts):
        inputs = texts
        data = [[0.1, 0.2, 0.3] for i in range(len(inputs))]
        return data, {"model": "my_model", "usage": {"total_tokens": 4, "prompt_tokens": 4}}

    def generate(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        return (
            ["This is a mocked response."],
            [{"role": "assistant", "usage": {"prompt_tokens": 5, "total_tokens": 10, "completion_tokens": 5}}],
        )
