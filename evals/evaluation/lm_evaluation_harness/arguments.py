#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2020 EleutherAI
# SPDX-License-Identifier: MIT

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from functools import partial


def _int_or_none_list_arg_type(max_len: int, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items != max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """Check to make sure all CLI args are typed, raises error if not."""
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=dict,
        default=None,
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`."),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3),
        default="0,1234,1234",  # for backward compatibility
        help=(
            "Set seed for python's random, numpy and torch.\n"
            "Accepts a comma-separated list of 3 values for python's random, numpy, and torch seeds, respectively, "
            "or a single integer to set the same seed for all three.\n"
            "The values are either an integer or 'None' to not set the seed. Default is `0,1234,1234` (for backward compatibility).\n"
            "E.g. `--seed 0,None,8` sets `random.seed(0)` and `torch.manual_seed(8)`. Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all three seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    check_argument_types(parser)
    return parser.parse_args()


class LMEvalParser:
    """The class is the another form of `setup_parser` function and used for function call pass parameters."""

    def __init__(
        self,
        model="hf",
        tasks="lambada_openai",
        model_args="",
        user_model=None,
        tokenizer=None,
        num_fewshot=None,
        batch_size=1,
        max_batch_size=None,
        device=None,
        output_path=None,
        limit=None,
        use_cache=None,
        cache_requests=None,
        check_integrity=False,
        write_out=False,
        log_samples=False,
        show_config=False,
        include_path=None,
        gen_kwargs=None,
        verbosity="INFO",
        wandb_args="",
        predict_only=False,
        seed=[0, 1234, 1234],
        trust_remote_code=False,
    ):
        self.model = model
        self.tasks = tasks
        self.model_args = model_args
        self.user_model = user_model
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.device = device
        self.output_path = output_path
        self.limit = limit
        self.use_cache = use_cache
        self.cache_requests = cache_requests
        self.check_integrity = check_integrity
        self.write_out = write_out
        self.log_samples = log_samples
        self.show_config = show_config
        self.include_path = include_path
        self.gen_kwargs = gen_kwargs
        self.verbosity = verbosity
        self.wandb_args = wandb_args
        self.predict_only = predict_only
        self.seed = seed
        self.trust_remote_code = trust_remote_code
