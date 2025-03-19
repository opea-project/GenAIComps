# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class OptunaArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    optuna: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use optuna"},
    )
    
    n_trials: int = field(
        default=30,
        metadata={"help": "Train bs"},
    )
    n_warmup_steps: int = field(
        default=15,
        metadata={"help": "Train bs"},
    )
    sampler: str = field(
        default="TPESampler",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    opt_params: str = field(
        default=None,
        metadata={"help": "Path to the folder containing the datasets."},
    )

    
    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

