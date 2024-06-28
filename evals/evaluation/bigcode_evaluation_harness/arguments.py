#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 BigCode Project
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import fnmatch

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.tasks import ALL_TASKS
from transformers import HfArgumentParser


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def setup_parser():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchmarks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memory to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    return parser.parse_args()


class BigcodeEvalParser:
    """The class is the another form of `setup_parser` function and used for function call pass parameters."""

    def __init__(
        self,
        prefix="",  # EvalArguments
        do_sample=True,
        temperature=0.2,
        top_k=0,
        top_p=0.95,
        n_samples=1,
        eos="<|endoftext|>",
        seed=0,
        model="codeparrot/codeparrot-small",  # BigcodeEvalArguments
        modeltype="causal",
        peft_model=None,
        revision=None,
        use_auth_token=False,
        trust_remote_code=False,
        tasks=None,
        instruction_tokens=None,
        batch_size=1,
        max_length_generation=512,
        precision="fp32",
        load_in_8bit=False,
        load_in_4bit=False,
        left_padding=False,
        limit=None,
        limit_start=0,
        save_every_k_tasks=-1,
        postprocess=True,
        allow_code_execution=False,
        generation_only=False,
        load_generations_path=None,
        load_data_path=None,
        metric_output_path="evaluation_results.json",
        save_generations=False,
        load_generations_intermediate_paths="",
        save_generations_path="generations.json",
        save_references=False,
        save_references_path="references.json",
        prompt="prompt",
        max_memory_per_gpu=None,
        check_references=False,
        user_model=None,  # used for pass model object
        tokenizer=None,  # used for pass tokenizer object
    ):
        self.prefix = prefix
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_samples = n_samples
        self.eos = eos
        self.seed = seed
        self.model = model
        self.modeltype = modeltype
        self.peft_model = peft_model
        self.revision = revision
        self.use_auth_token = use_auth_token
        self.trust_remote_code = trust_remote_code
        self.tasks = tasks
        self.instruction_tokens = instruction_tokens
        self.batch_size = batch_size
        self.max_length_generation = max_length_generation
        self.precision = precision
        self.load_int_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.left_padding = left_padding
        self.limit = limit
        self.limit_start = limit_start
        self.save_every_k_tasks = save_every_k_tasks
        self.postprocess = postprocess
        self.allow_code_execution = allow_code_execution
        self.generation_only = generation_only
        self.load_generations_path = load_generations_path
        self.load_data_path = load_data_path
        self.metric_output_path = metric_output_path
        self.save_generations = save_generations
        self.load_generations_intermediate_paths = load_generations_intermediate_paths
        self.save_generations_path = save_generations_path
        self.save_references = save_references
        self.save_references_path = save_references_path
        self.prompt = prompt
        self.max_memory_per_gpu = max_memory_per_gpu
        self.check_references = check_references
        self.user_model = user_model
        self.tokenizer = tokenizer
