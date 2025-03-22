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
from typing import Optional


# UI传入的参数，最后都会传到AdaclipArguments
# 后续在workflow.py里会根据这个把参数传到adaclip真正执行的main函数里
@dataclass
class AdaclipArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    resume: Optional[str] = field(
        default=None,
        metadata={"help": "dataset name"},
    )
    train_annot: Optional[str] = field(
        default=None,
        metadata={"help": "json file containing training video annotations"},
    )
    val_annot: Optional[str] = field(
        default=None,
        metadata={"help": "json file containing validation video annotations"},
    )
    test_annot: Optional[str] = field(
        default=None,
        metadata={"help": "json file containing test video annotations"},
    )
    frames_dir: Optional[str] = field(
        default=None,
        metadata={"help": "path to video frames"},
    )
    config: Optional[str] = field(
        default=None,
        metadata={"help": "config file path"},
    )
    adaclip_top_k: int = field(
        default=16,
        metadata={"help": "select top K frames in a video"},
    )
    adaclip_num_frm: int = field(
        default=2,
        metadata={"help": "frames to use per video"},
    )
    adaclip_batch_size: int = field(
        default=32,
        metadata={"help": "single-GPU batch size."},
    )
    val_batch_size: int = field(
        default=500,
        metadata={"help": "total # of training epochs."},
    )
    coef_lr: float = field(
        default=1e-3,
        metadata={"help": "lr multiplier for clip branch"},
    )
    warmup_proportion: float = field(
        default=0.1,
        metadata={
            "help": "proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training"
        },
    )
    init_tau: float = field(
        default=5.0,
        metadata={"help": "annealing init temperature"},
    )
    min_tau: float = field(
        default=0.5,
        metadata={"help": "min temperature to anneal to"},
    )
    adaclip_xpu: str = field(
        default=None,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    freeze_cnn: str = field(
        default=None,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    frame_agg: str = field(
        default=None,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg
