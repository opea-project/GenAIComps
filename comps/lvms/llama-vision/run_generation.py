#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Conditional text generation on Habana Gaudi/Gaudi2.
"""

import os
import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm

THIS_DIR = Path(__file__).parent.resolve()
import os, sys
print(str(THIS_DIR / "optimum"))
sys.path.append(str(THIS_DIR / "optimum"))

import torch
from utils import initialize_model

from transformers import MllamaForConditionalGeneration, AutoProcessor, AutoConfig, AutoModelForCausalLM
import habana_frameworks.torch.hpu as torch_hpu

from optimum.habana.utils import get_hpu_memory_stats




logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=25, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--num_images", type=int, default=1, help="number of images.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument("--user_prompt", type=str, default="Describe all of the images briefly.", help="User prompt.")
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--dataset_name",
        default='llavabench',
        type=str,
        help="Optional argument if you want to assess your model on a given dataset of the HF Hub.",
    )
    parser.add_argument(
        "--column_name",
        default=None,
        type=str,
        help="If `--dataset_name` was given, this will be the name of the column to use as prompts for generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling for generation.",
    )
    parser.add_argument(
        "--num_beams",
        default=1,
        type=int,
        help="Number of beams used for beam search generation. 1 means greedy search will be performed.",
    )
    parser.add_argument(
        "--top_k",
        default=None,
        type=int,
        help="Size of candidate set used for re-ranking in contrastive search. top_k > 1 enables contrastive search.",
    )
    parser.add_argument(
        "--penalty_alpha",
        default=None,
        type=float,
        help="Degeneration penalty for contrastive search. penalty_alpha > 0 enables contrastive search.",
    )
    parser.add_argument(
        "--trim_logits",
        action="store_true",
        help="Calculate logits only for the last token to save memory in the first step.",
    )
    parser.add_argument(
        "--seed",
        default=27,
        type=int,
        help="Seed to use for random generation. Useful to reproduce your runs with `--do_sample`.",
    )
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--profiling_record_shapes",
        default=False,
        type=bool,
        help="Record shapes when enabling profiling.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        type=str,
        nargs="*",
        help='Optional argument to give a prompt of your choice as input. Can be a single string (eg: --prompt "Hello world"), or a list of space-separated strings (eg: --prompt "Hello world" "How are you?")',
    )
    parser.add_argument(
        "--bad_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that are not allowed to be generated.",
    )
    parser.add_argument(
        "--force_words",
        default=None,
        type=str,
        nargs="+",
        help="Optional argument list of words that must be generated.",
    )
    parser.add_argument(
        "--assistant_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a draft/assistant model for assisted decoding.",
    )
    parser.add_argument(
        "--peft_model",
        default=None,
        type=str,
        help="Optional argument to give a path to a PEFT model.",
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument(
        "--token",
        default=None,
        type=str,
        help="The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
        "generated when running `huggingface-cli login` (stored in `~/.huggingface`).",
    )
    parser.add_argument(
        "--model_revision",
        default="main",
        type=str,
        help="The specific model version to use (can be a branch name, tag name or commit id).",
    )
    parser.add_argument(
        "--attn_softmax_bf16",
        action="store_true",
        help="Whether to run attention softmax layer in lower precision provided that the model supports it and "
        "is also running in lower precision.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output directory to store results in.",
    )
    parser.add_argument(
        "--bucket_size",
        default=-1,
        type=int,
        help="Bucket size to maintain static shapes. If this number is negative (default is -1) \
            then we use `shape = prompt_length + max_new_tokens`. If a positive number is passed \
            we increase the bucket in steps of `bucket_size` instead of allocating to max (`prompt_length + max_new_tokens`).",
    )
    parser.add_argument(
        "--bucket_internal",
        action="store_true",
        help="Split kv sequence into buckets in decode phase. It improves throughput when max_new_tokens is large.",
    )
    parser.add_argument(
        "--dataset_max_samples",
        default=-1,
        type=int,
        help="If a negative number is passed (default = -1) perform inference on the whole dataset, else use only `dataset_max_samples` samples.",
    )
    parser.add_argument(
        "--limit_hpu_graphs",
        action="store_true",
        help="Skip HPU Graph usage for first token to save memory",
    )
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Whether to reuse key/value cache for decoding. It should save memory.",
    )
    parser.add_argument("--verbose_workers", action="store_true", help="Enable output from non-master workers")
    parser.add_argument(
        "--simulate_dyn_prompt",
        default=None,
        type=int,
        nargs="*",
        help="If empty, static prompt is used. If a comma separated list of integers is passed, we warmup and use those shapes for prompt length.",
    )
    parser.add_argument(
        "--reduce_recompile",
        action="store_true",
        help="Preprocess on cpu, and some other optimizations. Useful to prevent recompilations when using dynamic prompts (simulate_dyn_prompt)",
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Whether to enable Habana Flash Attention, provided that the model supports it.",
    )
    parser.add_argument(
        "--flash_attention_recompute",
        action="store_true",
        help="Whether to enable Habana Flash Attention in recompute mode on first token generation. This gives an opportunity of splitting graph internally which helps reduce memory consumption.",
    )
    parser.add_argument(
        "--flash_attention_causal_mask",
        action="store_true",
        help="Whether to enable Habana Flash Attention in causal mode on first token generation.",
    )
    parser.add_argument(
        "--flash_attention_fast_softmax",
        action="store_true",
        help="Whether to enable Habana Flash Attention in fast softmax mode.",
    )
    parser.add_argument(
        "--book_source",
        action="store_true",
        help="Whether to use project Guttenberg books data as input. Usefull for testing large sequence lenghts.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Whether to use torch compiled model or not.",
    )
    parser.add_argument(
        "--ignore_eos",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to disable stopping with eos token when calling `generate`. --no-ignore_eos to disable it",
    )
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature value for text generation")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top_p value for generating text via sampling")
    parser.add_argument(
        "--const_serialization_path",
        "--csp",
        type=str,
        help="Path to serialize const params. Const params will be held on disk memory instead of being allocated on host memory.",
    )
    parser.add_argument(
        "--disk_offload",
        action="store_true",
        help="Whether to enable device map auto. In case no space left on cpu, weights will be offloaded to disk.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust the execution of code from datasets/models defined on the Hub. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.",
    )
    parser.add_argument(
        "--load_quantized_model",
        action="store_true",
        help="Whether to load model from hugging face checkpoint.",
    )
    parser.add_argument(
        "--parallel_strategy",
        type=str,
        choices=["tp", "none"],  # Add other strategies as needed
        default="none",
        help="Run multi card with the specified parallel strategy. Choices are 'tp' for Tensor Parallel Strategy or 'none'.",
    )

    args = parser.parse_args()

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    if args.use_flash_attention and not args.flash_attention_fast_softmax:
        args.flash_attention_fast_softmax = True

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config == "" and args.disk_offload:
        logger.warning(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )
    return args


def load_dataset_with_name(dataset, n_iterations, num_images, user_prompt="Describe all of the images briefly."):
    if dataset == "llavabench":
        from datasets import load_dataset, Image
        ds = load_dataset("liuhaotian/llava-bench-in-the-wild", split="train")
        ds = ds.cast_column("image", Image(mode="RGB"))
        ds = ds.add_column("prompt", [f'<|image|><|begin_of_text|>{user_prompt}']*len(ds))
    if dataset == "vqa_1":
        from datasets import load_dataset, Image
        ds = load_dataset("Graphcore/vqa", split=f"train[:{n_iterations}]", trust_remote_code=True)
        ds = ds.cast_column("image_id", Image(mode="RGB"))
        ds = ds.map(lambda x: {"prompt": "<|image|><|begin_of_text|>" + x["question"]})
        ds = ds.rename_columns({"image_id": "image"})
    if dataset == "vqa":
        from datasets import load_dataset, Image
        ds = load_dataset("Graphcore/vqa", split=f"train[:{n_iterations}]", trust_remote_code=True)
        ds = ds.cast_column("image_id", Image(mode="RGB"))
        ds = ds.add_column("prompt", [f'<|image|><|begin_of_text|>{user_prompt}']*len(ds))
        ds = ds.rename_columns({"image_id": "image"})
    if dataset == "vqa_example":
        from datasets import Dataset
        import requests
        from PIL import Image
        raw_image = Image.open(requests.get("https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg", stream=True).raw)
        user_prompt = 'how many dogs are in the picture?'
        ds = Dataset.from_dict({'image': [raw_image] * n_iterations, 'prompt': [f'<|image|><|begin_of_text|>{user_prompt}'] * n_iterations})
    if dataset == "llava_example":
        from datasets import Dataset
        import requests
        from PIL import Image
        raw_image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        user_prompt = 'If I had to write a haiku for this one'
        ds = Dataset.from_dict({'image': [raw_image] * n_iterations, 'prompt': [f'<|image|><|begin_of_text|>{user_prompt}'] * n_iterations})
    return ds


def run(args, model, input_processor, tokenizer,  use_lazy_mode, generation_config):
    # Downloading and loading a dataset from the hub.
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    raw_dataset = load_dataset_with_name(args.dataset_name, args.batch_size*args.num_images, args.num_images, user_prompt=args.user_prompt)
    
    def preprocess_function(examples):
        return input_processor(
            examples['prompt'],
            examples['image'],
            return_tensors="pt",
        )

    raw_dataset = raw_dataset.map(
        preprocess_function,
        desc="Running input_processor on dataset",
    )
    
    raw_dataset = raw_dataset.select_columns(['prompt', 'input_ids', 'attention_mask', 'pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask', 'cross_attention_mask'])
    raw_dataset.set_format(type="torch")

    def collate_fn(batch):
        collect = {}
        for data in batch:
            for k, tensors in data.items():
                if k not in collect:
                    collect[k] = []
                collect[k].append(tensors)
        for k in collect:
            tensors = collect[k]
            if torch.is_tensor(tensors[0]):
                max_shape = max([item.shape[1] for item in tensors])
                min_shape = min([item.shape[1] for item in tensors])
                if max_shape != min_shape:
                    # pad for the batch
                    if len(tensors[0].shape) == 2:
                        padded_tensors = [torch.cat((torch.zeros((1, max_shape - item.shape[1]), dtype=item.dtype), item), 1) for item in tensors]
                    else:
                        padded_tensors = [torch.concat((torch.zeros((1, max_shape - item.shape[1], *item.shape[2:]), dtype=item.dtype), item), 1) for item in tensors]
                else:
                    padded_tensors = tensors
                collect[k] = torch.concat(padded_tensors, 0)

        return collect
            
    #print("raw_dataset is ", raw_dataset, "batch size is ", args.batch_size)
    dataloader = DataLoader(raw_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    def generate(batch):
        # Generate new sequences
        prompt = batch['prompt']
        input_tokens = batch['input_ids']
        batch.pop('prompt')
        for t in batch:
            if torch.is_tensor(batch[t]):
                batch[t] = batch[t].to(model.device)
        outputs = model.generate(
            **batch,
            generation_config=generation_config,
            do_sample=False,
            max_new_tokens=args.max_new_tokens, 
            pad_token_id=args.pad_token_id,
            #ignore_eos=True,
        )
        if isinstance(outputs, tuple):
            outputs, first_token_latency, next_token_avg_latency = outputs[0], outputs[1], outputs[2]
        else:
            outputs = outputs
            first_token_latency = 0
            next_token_avg_latency = 0
        outputs = outputs.cpu()
        return prompt, input_tokens, outputs, first_token_latency, next_token_avg_latency

    import habana_frameworks.torch as htorch
    htorch.hpu.ModuleCacher()(model=model, inplace=True)
    # warmup
    if args.warmup>0:
        from optimum.habana.utils import HabanaProfile

        # compilation stage disable profiling
        HabanaProfile.disable()
        # Compilation
        logger.info("Graph compilation...")
        t0 = time.perf_counter()
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="graph warmup"):
            print("batch shape is ", [(k, (v.shape if torch.is_tensor(v) else len(v))) for k, v in batch.items()], flush=True)
            generate(batch)
            # The first three iterations take longer because of graph compilation
            if (i + 1) == 3:
                break
        torch_hpu.synchronize()
        compilation_duration = time.perf_counter() - t0
        HabanaProfile.enable()

    total_input_tokens = 0
    total_new_tokens_generated = 0
    num_requests = 0
    duration = 0
    duration_list = []
    first_token_latency_list = []
    next_token_avg_latency_list = []
    separator = "-" * 50
    logger.info("Running generate dataset...")
    t_start = time.time()
    for i in tqdm(range(args.n_iterations), total=args.n_iterations, desc="Benchmarking"):
        for _, batch in enumerate(dataloader):
            t0 = time.perf_counter()
            prompt, input_tokens, outputs, first_token_latency, next_token_avg_latency = generate(batch)
            elapsed_time = time.perf_counter() - t0
            duration += elapsed_time
            duration_list.append(elapsed_time)
            first_token_latency_list.append(first_token_latency)
            next_token_avg_latency_list.append(next_token_avg_latency)
            num_requests += len(prompt)
            input_tokens = sum([len(i) for i in input_tokens])
            total_input_tokens += input_tokens
            total_new_tokens_generated += (sum([len(o) for o in outputs]) - input_tokens)
        print(separator)
        print(f"Batch n°{i+1}")
        print(f"Input: {prompt[:args.batch_size]}")
        print(
            f"Output: {tokenizer.batch_decode(outputs, skip_special_tokens=True)[:args.batch_size*args.num_return_sequences]}"
        )
        print(separator)
    t_end = time.time()

    in_throughput = total_input_tokens / duration
    out_throughput = total_new_tokens_generated / duration
    request_throughput = num_requests / duration
    # Print Stats
    config_str = f"Batch size = {args.batch_size}\n"
    config_str += f"Total n_iteration is {args.n_iterations}\n"
    config_str += f"max_new_tokens is {args.max_new_tokens}\n"
    stats = f"Average Latency = {(sum(duration_list) / len(duration_list))} seconds\n"
    stats += f"Average first token Latency = {(sum(first_token_latency_list) / len(first_token_latency_list))} seconds\n"
    stats += f"Average next token Latency = {(sum(next_token_avg_latency_list) / len(next_token_avg_latency_list))} seconds\n"
    stats += f"Input Throughput = {in_throughput} tokens/second\n"
    stats += f"Output Throughput = {out_throughput} tokens/second\n"
    stats += f"Image Requests Throughput = {request_throughput} requests/second\n"
    separator = "-" * 50
    print()
    print("Benchmark config:")
    print(separator)
    print(config_str)
    print()
    print("Stats:")
    print(separator)
    print(stats)
    print("Total runtime for dataset:", t_end - t_start)
    mem = get_hpu_memory_stats()
    for k, v in mem.items():
        print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))
    if args.warmup > 0:
        print(f"Graph compilation duration          = {compilation_duration} seconds")
    print(separator)


def main():
    parser = argparse.ArgumentParser()
    args = setup_parser(parser)

    model, assistant_model, tokenizer, generation_config = initialize_model(args, logger)
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>> generation_config = {generation_config}")
    input_processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    print(f">>>>>>>>>>>>>>>>>>>>>>>>>> model = {model}")
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>> model device is = {model.device}")
    import subprocess
    print(subprocess.check_output(["hl-smi"]).decode("utf-8"))
    args.pad_token_id = tokenizer.pad_token_id
    print("pad token id is ", args.pad_token_id)

    use_lazy_mode = True
    if args.torch_compile:
        use_lazy_mode = False

    run(args, model, input_processor, tokenizer, use_lazy_mode, generation_config)

if __name__ == "__main__":
    main()

