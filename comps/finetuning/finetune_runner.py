# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import time
import uuid
from typing import List

from llm_on_ray.finetune.finetune_config import FinetuneConfig
from pydantic_yaml import parse_yaml_raw_as
from ray.train.base_trainer import TrainingFailedError
from ray.tune.callback import Callback
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class FineTuneCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("FineTuneCallback:", args, state)


def main():
    parser = argparse.ArgumentParser(description="Runner for llm_on_ray-finetune")
    parser.add_argument("--config_file", type=str, required=True, default=None)
    args = parser.parse_args()
    model_config_file = args.config_file

    with open(model_config_file) as f:
        finetune_config = parse_yaml_raw_as(FinetuneConfig, f).model_dump()

    callback = FineTuneCallback()
    finetune_config["Training"]["callbacks"] = [callback]

    from llm_on_ray.finetune.finetune import main as llm_on_ray_finetune_main

    llm_on_ray_finetune_main(finetune_config)
    # try:
    #     llm_on_ray_finetune_main(finetune_config)
    # except TrainingFailedError as e:
    #     print(e)


if __name__ == "__main__":
    main()
