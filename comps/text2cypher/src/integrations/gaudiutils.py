# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import glob
import os
import shutil
import tempfile
import time
from pathlib import Path

import requests
import torch
from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    get_repo_root,
    model_is_optimized,
    model_on_meta,
    write_checkpoints_json,
)
from optimum.habana.utils import (
    check_habana_frameworks_version,
    check_optimum_habana_min_version,
    get_habana_frameworks_version,
    set_seed,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import check_min_version

from comps import CustomLogger

logger = CustomLogger("opea_text2cypher_utils")


def setup_parser(parser):
    # Arguments management
    parser.add_argument("--device", "-d", type=str, choices=["hpu"], help="Device to run", default="hpu")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        # required=True,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--bf16",
        default=True,
        action="store_true",
        help="Whether to perform generation in bf16 precision.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate.")
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=0,
        help="If > 0 then pad and truncate the input sequences to this specified length of tokens. \
            if == 0, then truncate to 16 (original default) \
            if < 0, then do not truncate, use full input prompt",
    )
    parser.add_argument("--tokenizer-config-path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=1, help="Input batch size.")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations for benchmarking.")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of inference iterations for benchmarking.")
    parser.add_argument("--local_rank", type=int, default=0, metavar="N", help="Local process rank.")
    parser.add_argument(
        "--use_kv_cache",
        default=True,
        action="store_true",
        help="Whether to use the key/value cache for decoding. It should speed up generation.",
    )
    parser.add_argument(
        "--use_hpu_graphs",
        default=True,
        action="store_true",
        help="Whether to use HPU graphs or not. Using HPU graphs should give better latencies.",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
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
        help="Whether to use project Guttenberg books data as input. Useful for testing large sequence lengths.",
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
        help="Whether to ignore eos, set False to disable it",
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
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    args = parser.parse_args()

    if args.torch_compile:
        args.use_hpu_graphs = False

    if not args.use_hpu_graphs:
        args.limit_hpu_graphs = False

    args.quant_config = os.getenv("QUANT_CONFIG", "")
    if args.quant_config == "" and args.disk_offload:
        logger.warning(
            "`--disk_offload` was tested only with fp8, it may not work with full precision. If error raises try to remove the --disk_offload flag."
        )
    return args


def adjust_batch(batch, size):
    curr_size = batch["input_ids"].shape[1]
    if curr_size >= size:
        adjusted_batch = {
            "input_ids": batch["input_ids"][:, :size],
            "attention_mask": batch["attention_mask"][:, :size],
        }
    else:
        adjusted_batch = {}
        for k in batch.keys():
            last_colm = batch[k][:, -1]
            expanded = last_colm.tile((size - curr_size, 1)).T
            adjusted_batch[k] = torch.concat([batch[k], expanded], 1)
    assert adjusted_batch["input_ids"].shape[1] == size
    assert adjusted_batch["attention_mask"].shape[1] == size
    return adjusted_batch


def override_print(enable):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def override_logger(logger, enable):
    logger_info = logger.info

    def info(*args, **kwargs):
        force = kwargs.pop("force", False)
        if force or enable:
            logger_info(*args, **kwargs)

    logger.info = info


def count_hpu_graphs():
    return len(glob.glob(".graph_dumps/*PreGraph*"))


def override_prints(enable, logger):
    override_print(enable)
    override_logger(logger, enable)


def setup_inference(args, model):
    import habana_frameworks.torch.core as htcore

    habana_version = get_habana_frameworks_version()

    print("Initializing inference mode")
    # Keeping the if-else here for back compat. TODO remove later
    if habana_version.major >= 1 and habana_version.minor >= 16:
        htcore.hpu_initialize(model, mark_only_scales_as_const=True)
    else:
        const_marking = os.getenv("ENABLE_CONST_MARKING", "True")
        if const_marking == "True":
            htcore.hpu_initialize(model)
    return model


def setup_const_serialization(const_serialization_path):
    import uuid

    const_serialization_path = os.path.join(const_serialization_path + uuid.uuid4().hex)
    os.makedirs(const_serialization_path)
    from habana_frameworks.torch.hpu import enable_const_section_serialization

    print("Serializing const params to {}".format(const_serialization_path))
    enable_const_section_serialization(const_serialization_path, True)


def setup_env(args):
    # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
    check_min_version("4.34.0")
    check_optimum_habana_min_version("1.9.0.dev0")
    # TODO: SW-167588 - WA for memory issue in hqt prep_model
    os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")

    if args.use_hpu_graphs and args.limit_hpu_graphs and not args.reuse_cache and args.bucket_internal:
        # Based upon above conditions and below env variable,
        # we can call HPU graphs clear_inputs().
        os.environ.setdefault("PT_HPUGRAPH_DISABLE_TENSOR_CACHE", "1")

    # Tweak generation so that it runs faster on Gaudi
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    adapt_transformers_to_gaudi()


def setup_device(args):
    if args.device == "hpu":
        import habana_frameworks.torch.core as htcore

        if args.quant_config:
            htcore.hpu_set_env()
    return torch.device(args.device)


# patching LinearAllreduce to use ScopedLinearAllReduce
def patch_scoped_linear_all_reduce(model):
    from deepspeed.module_inject.layers import LinearAllreduce
    from optimum.habana.transformers.models.modeling_all_models import ScopedLinearAllReduce

    for name, module in model.named_children():
        if type(module) is LinearAllreduce:
            SL = ScopedLinearAllReduce(mod=module)
            setattr(model, name, SL)
        patch_scoped_linear_all_reduce(module)


def get_torch_compiled_model(model):
    model.model = torch.compile(model.model, backend="hpu_backend", options={"keep_input_mutations": True})
    return model


def setup_model(args, model_dtype, model_kwargs, logger):
    logger.info("Single-device run.")
    os.environ["HUGGINGFACE_HUB_TIMEOUT"] = "300"
    if args.assistant_model is None:
        assistant_model = None
    else:
        logger.info(f"Using asssitant model {args.assistant_model}.")
    if args.disk_offload:
        from accelerate import infer_auto_device_map, init_empty_weights

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        max_memory = {"cpu": "10GiB"}
        device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=model_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map=device_map,
            offload_folder="/tmp/offload_folder/",
            offload_state_dict=True,
            torch_dtype=model_dtype,
            **model_kwargs,
        )
    else:
        if args.assistant_model is not None:
            assistant_model = AutoModelForCausalLM.from_pretrained(
                args.assistant_model, torch_dtype=model_dtype, **model_kwargs
            )
        if args.peft_model is not None:
            model = peft_model(args, model_dtype, logger, **model_kwargs)
        else:
            if args.model_name_or_path == "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1":
                model = AutoModelForCausalLM.from_pretrained(
                    "google/gemma-2-9b-it", torch_dtype=model_dtype, **model_kwargs
                )
                model.load_adapter(args.model_name_or_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs
                )
    if args.quant_config:
        import habana_quantization_toolkit

        habana_quantization_toolkit.prep_model(model)
        if args.assistant_model is not None:
            habana_quantization_toolkit.quantize_model(assistant_model)

    model = model.eval().to(args.device)
    if args.assistant_model is not None:
        assistant_model = assistant_model.eval().to(args.device)

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        from optimum.habana.transformers.trainer import _is_peft_model

        if check_habana_frameworks_version("1.13.0") and model.config.model_type == "falcon":
            model = wrap_in_hpu_graph(model, hash_with_views=False)
        else:
            model = wrap_in_hpu_graph(model)
        if args.assistant_model is not None:
            assistant_model = wrap_in_hpu_graph(assistant_model)
        if _is_peft_model(model):
            model.base_model = wrap_in_hpu_graph(model.base_model)

    if args.torch_compile and model.config.model_type == "llama":
        model = get_torch_compiled_model(model)
        # if args.assistant_model is not None:
        #     assistant_model = get_torch_compiled_model(assistant_model)
    return model, assistant_model


def peft_model(args, model_dtype, logger, **model_kwargs):
    import importlib.util

    if importlib.util.find_spec("peft") is None:
        raise ImportError("The `peft` package is not installed, please run: `pip install peft`.")
    from peft import AutoPeftModelForCausalLM
    from peft.config import PeftConfigMixin

    base_model_name = PeftConfigMixin.from_pretrained(
        args.peft_model,
        token=model_kwargs["token"] if "token" in model_kwargs else None,
    ).base_model_name_or_path

    base_model_is_local = Path(base_model_name).is_dir()
    if not base_model_is_local:
        # Check if the base model path to a remote repository on the HF Hub exists
        from huggingface_hub import list_repo_files

        try:
            list_repo_files(base_model_name)
            base_model_is_remote = True
        except Exception:
            base_model_is_remote = False

    if base_model_is_local or base_model_is_remote:
        model = AutoPeftModelForCausalLM.from_pretrained(args.peft_model, torch_dtype=model_dtype, **model_kwargs)
    else:
        # Since the base model doesn't exist locally nor remotely, use `args.model_name_or_path` as the base model
        logger.warning(
            f"The base model `{base_model_name}` of the LoRA configuration associated"
            f" to `{args.peft_model}` does not exist locally or remotely. Using "
            f"`--model_name_or_path {args.model_name_or_path}` as a fall back for the base model."
        )
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=model_dtype, **model_kwargs)
        model = PeftModel.from_pretrained(model, args.peft_model, torch_dtype=model_dtype, **model_kwargs)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        if model_dtype == torch.bfloat16:
            model = model.to(torch.bfloat16)
        return model
    else:
        from optimum.habana.peft.peft_model import gaudi_generate, gaudi_prepare_inputs_for_generation

        model.__class__.generate = gaudi_generate
        model.__class__.prepare_inputs_for_generation = gaudi_prepare_inputs_for_generation
        return model


def setup_tokenizer(args, model, assistant_model):
    tokenizer_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
        "chat_template": "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}",
    }
    if args.bad_words is not None or args.force_words is not None:
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"

    if model.config.model_type == "llama":
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        if assistant_model is not None:
            assistant_model.generation_config.pad_token_id = 0
            assistant_model.generation_config.bos_token_id = 1
            assistant_model.generation_config.eos_token_id = 2
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)
    if model.config.model_type == "persimmon":
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if assistant_model is not None:
            assistant_model.generation_config.pad_token_id = assistant_model.generation_config.eos_token_id
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if assistant_model is not None:
            assistant_model.generation_config.pad_token_id = assistant_model.generation_config.eos_token_id

    return tokenizer, model, assistant_model


def setup_generation_config(args, model, assistant_model, tokenizer):
    bad_words_ids = None
    force_words_ids = None
    if args.bad_words is not None:
        bad_words_ids = [tokenizer.encode(bad_word, add_special_tokens=False) for bad_word in args.bad_words]
    if args.force_words is not None:
        force_words_ids = [tokenizer.encode(force_word, add_special_tokens=False) for force_word in args.force_words]

    is_optimized = model_is_optimized(model.config)

    # Generation configuration
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = args.use_kv_cache
    generation_config.static_shapes = is_optimized and assistant_model is None
    generation_config.bucket_size = args.bucket_size if is_optimized else -1
    generation_config.bucket_internal = args.bucket_internal
    generation_config.do_sample = args.do_sample
    generation_config.num_beams = args.num_beams
    generation_config.bad_words_ids = bad_words_ids
    generation_config.force_words_ids = force_words_ids
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.trim_logits = args.trim_logits
    generation_config.attn_softmax_bf16 = args.attn_softmax_bf16
    generation_config.limit_hpu_graphs = args.limit_hpu_graphs
    generation_config.reuse_cache = args.reuse_cache
    generation_config.reduce_recompile = args.reduce_recompile
    if generation_config.reduce_recompile:
        assert generation_config.bucket_size > 0
    generation_config.use_flash_attention = args.use_flash_attention
    generation_config.flash_attention_recompute = args.flash_attention_recompute
    generation_config.flash_attention_causal_mask = args.flash_attention_causal_mask
    generation_config.flash_attention_fast_softmax = args.flash_attention_fast_softmax
    generation_config.trust_remote_code = args.trust_remote_code

    return generation_config


def exclude_hpu_graph_configs(args):
    # Excluded configs for batch size 1 for hpu graph
    if args.batch_size == 1 and args.limit_hpu_graphs:
        if "falcon-180B" in args.model_name_or_path or "falcon-180b" in args.model_name_or_path:
            return False
        return True
    else:
        return False


def initialize_model(args, logger):
    init_start = time.perf_counter()
    if exclude_hpu_graph_configs(args):
        args.limit_hpu_graphs = False
    override_prints(args.verbose_workers, logger)
    setup_env(args)
    setup_device(args)
    set_seed(args.seed)
    get_repo_root(args.model_name_or_path, local_rank=0, token=args.token)
    if args.assistant_model is not None:
        get_repo_root(args.assistant_model, local_rank=0, token=args.token)
    use_deepspeed = False
    if use_deepspeed or args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float
        args.attn_softmax_bf16 = False
    model_kwargs = {
        "revision": args.model_revision,
        "token": args.token,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.trust_remote_code:
        logger.warning("`trust_remote_code` is set, there is no guarantee this model works properly and it may fail")

    model, assistant_model = setup_model(args, model_dtype, model_kwargs, logger)
    tokenizer, model, assistant_model = setup_tokenizer(args, model, assistant_model)
    generation_config = setup_generation_config(args, model, assistant_model, tokenizer)

    if args.const_serialization_path:
        setup_const_serialization(args.const_serialization_path)
    if args.quant_config:
        model = setup_inference(args, model)
    init_end = time.perf_counter()
    logger.info(f"Args: {args}")
    logger.info(f"device: {args.device}, n_hpu: 1, bf16: {model_dtype == torch.bfloat16}")
    logger.info(f"Model initialization took {(init_end - init_start):.3f}s")
    return model, assistant_model, tokenizer, generation_config
