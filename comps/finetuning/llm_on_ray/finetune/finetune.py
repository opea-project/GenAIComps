#
# Copyright 2023 The LLM-on-Ray Authors.
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
#

#!/usr/bin/env python

import argparse
import os
import sys
from importlib import util
from typing import Any, Dict, Optional, Union

import ray
import torch
import transformers
from llm_on_ray import common
from llm_on_ray.finetune.finetune_config import FinetuneConfig
from pydantic_yaml import parse_yaml_raw_as
from ray.air import FailureConfig, RunConfig
from ray.air.config import ScalingConfig
from ray.train.torch import TorchTrainer

use_habana = False
if util.find_spec("habana_frameworks") is not None:
    from optimum.habana.utils import set_seed

    use_habana = True
else:
    from accelerate.utils import is_xpu_available, set_seed

    use_habana = False


def get_accelerate_environment_variable(config: Dict[str, Any]) -> dict:
    device = config["Training"]["device"]
    accelerate_mode = config["Training"]["accelerate_mode"]
    mixed_precision = config["Training"]["mixed_precision"]
    mode_env_vars = {
        "cpu": {
            "DDP": {
                "ACCELERATE_USE_CPU": "true",
                "ACCELERATE_USE_IPEX": "true",
                "ACCELERATE_MIXED_PRECISION": mixed_precision,
            }
        },
        "gpu": {
            "DDP": {
                "ACCELERATE_USE_CPU": "false",
                "ACCELERATE_USE_XPU": "true",
                "ACCELERATE_USE_IPEX": "true",
                "ACCELERATE_MIXED_PRECISION": mixed_precision,
            },
            "FSDP": {
                "ACCELERATE_USE_CPU": "false",
                "ACCELERATE_USE_XPU": "true",
                "ACCELERATE_USE_IPEX": "true",
                "ACCELERATE_USE_FSDP": "true",
                "FSDP_SHARDING_STRATEGY": "1",
                "FSDP_OFFLOAD_PARAMS": "false",
                "FSDP_AUTO_WRAP_POLICY": "NO_WRAP",
                "FSDP_BACKWARD_PREFETCH": "BACKWARD_PRE",
                "FSDP_STATE_DICT_TYPE": "SHARDED_STATE_DICT",
                "FSDP_FORWARD_PREFETCH": "false",
                "FSDP_USE_ORIG_PARAMS": "false",
                "FSDP_SYNC_MODULE_STATES": "true",
                "ACCELERATE_MIXED_PRECISION": mixed_precision,
            },
            "DEEPSPEED": {
                "ACCELERATE_USE_CPU": "false",
                "ACCELERATE_USE_XPU": "true",
                "ACCELERATE_USE_IPEX": "true",
                "ACCELERATE_USE_DEEPSPEED": "true",
                "ACCELERATE_MIXED_PRECISION": mixed_precision,
            },
        },
        "hpu": {
            "DDP": {
                "ACCELERATE_USE_CPU": "false",
                "ACCELERATE_USE_XPU": "false",
                "ACCELERATE_USE_IPEX": "false",
                "ACCELERATE_MIXED_PRECISION": mixed_precision,
            },
            "DEEPSPEED": {
                "ACCELERATE_USE_CPU": "false",
                "ACCELERATE_USE_XPU": "false",
                "ACCELERATE_USE_IPEX": "false",
                "ACCELERATE_USE_DEEPSPEED": "true",
                "ACCELERATE_MIXED_PRECISION": mixed_precision,
            },
        },
    }
    if device not in mode_env_vars or accelerate_mode not in mode_env_vars[device]:
        supported_mode_info = ""
        for k in mode_env_vars.keys():
            supported_mode_info += k + ":["
            for m in mode_env_vars[k]:
                supported_mode_info += m + ","
            supported_mode_info = supported_mode_info[:-1]
            supported_mode_info += "],"
        supported_mode_info = supported_mode_info[:-1]

        raise ValueError(
            f"device {device} and accelerate mode {accelerate_mode} not supported. supported device and accelerate mode is {supported_mode_info}"
        )
    return mode_env_vars[device][accelerate_mode]


def convert_to_training_args(cls, config):
    device = config["Training"]["device"]
    accelerate_mode = config["Training"]["accelerate_mode"]
    save_strategy = config["General"]["save_strategy"]

    args = {
        "output_dir": config["General"]["output_dir"],
        "resume_from_checkpoint": config["General"]["resume_from_checkpoint"],
        "gradient_checkpointing": config["General"]["enable_gradient_checkpointing"],
        "save_strategy": save_strategy if save_strategy != "False" else "no",
        "bf16": config["Training"]["mixed_precision"] == "bf16",
        "num_train_epochs": config["Training"]["epochs"],
        "per_device_train_batch_size": config["Training"]["batch_size"],
        "per_device_eval_batch_size": config["Training"]["batch_size"],
        "optim": config["Training"]["optimizer"],
        "learning_rate": config["Training"]["learning_rate"],
        "logging_steps": config["Training"]["logging_steps"],
        "lr_scheduler_type": config["Training"]["lr_scheduler"],
        "weight_decay": config["Training"]["weight_decay"],
        "gradient_accumulation_steps": config["Training"]["gradient_accumulation_steps"],
    }

    # set attr max_steps
    if config["Training"]["max_train_steps"] is not None:
        args.update({"max_steps": config["Training"]["max_train_steps"]})

    # set attr for device cpu
    if device == "cpu":
        if hasattr(cls, "use_cpu"):
            args.update({"use_cpu": True})
        if hasattr(cls, "no_cuda"):
            args.update({"no_cuda": True})
        args.update({"use_ipex": True})

    # set attr 'deepspeed'
    if accelerate_mode == "DEEPSPEED":
        args.update({"deepspeed": config["Training"]["deepspeed_config_file"]})

    # set attr for FSDP
    # if accelerate_mode == "FSDP":
    #     args.updatwe({})

    # set attr for Intel Gaudi
    if device == "hpu":
        args.update({"use_habana": True})
        args.update({"use_lazy_mode": config["Training"]["hpu_execution_mode"] == "lazy"})
        args.update({"pipelining_fwd_bwd": True})

    return cls(**args)


def get_device_environment_variable(device):
    if device == "hpu":
        return {
            "HABANA_VISIBLE_DEVICES": "all",
            "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES": "true",
        }
    return {}


def convert_dtype(dtype: str) -> Optional[torch.dtype]:
    supported_dtypes = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "no": None,
    }
    return supported_dtypes[dtype]


def train_func(config: Dict[str, Any]):
    os.chdir(config["cwd"])

    device = config["Training"]["device"]

    base_model = config["General"]["base_model"]
    if config["General"].get("tokenizer_name") is not None:
        tokenizer_name = config["General"].get("tokenizer_name")
    else:
        tokenizer_name = base_model
    dataset_file = config["Dataset"]["train_file"]

    seed = config["Training"].get("seed")
    if seed is not None:
        set_seed(seed)

    tokenizer = common.tokenizer.Tokenizer.registory.get("HuggingFaceTokenizer")()(
        config={
            "name": tokenizer_name,
            "config": config["General"]["config"],
        }
    )

    datasets = common.dataset.Dataset.registory.get("HuggingfaceDataset")()(
        config={
            "name": dataset_file,
            "validation_file": config["Dataset"]["validation_file"],
            "validation_split_percentage": config["Dataset"]["validation_split_percentage"],
        }
    )

    dataprocesser = common.dataprocesser.DataProcesser.registory.get("GeneralProcesser")(
        config={
            "per_device_train_batch_size": config["Training"]["batch_size"],
            "per_device_eval_batch_size": config["Training"]["batch_size"],
            "preprocessing_num_workers": config["Dataset"].get("preprocessing_num_workers", 1),
            "max_length": config["Dataset"].get("max_length", 512),
            "group": config["Dataset"].get("group", True),
            "block_size": config["Dataset"].get("block_size", 512),
            "shuffle": config["Dataset"].get("shuffle", False),
        }
    )
    tokenized_datasets = dataprocesser.tokenize_dataset(tokenizer, datasets)

    model = common.model.Model.registory.get("HuggingFaceModelForCausalLM")()(
        config={
            "name": base_model,
            "dtype": convert_dtype(config["Training"].get("mixed_precision", "no")),
            "device": torch.device(device),
            "config": config["General"]["config"],
            "enable_gradient_checkpointing": config["General"].get("enable_gradient_checkpointing", False),
            "lora_config": config["General"].get("lora_config", None),
        }
    )

    data_collator = common.dataprocesser.general_processer.DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    if device in ["cpu", "gpu"]:
        from transformers import Trainer, TrainingArguments

        training_args = convert_to_training_args(TrainingArguments, config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        common.logger.info("train start")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        common.logger.info("train finish")
    elif device in ["hpu"]:
        from optimum.habana import GaudiConfig
        from optimum.habana.transformers import GaudiTrainer, GaudiTrainingArguments

        # If gaudi_config_name is provided, load gaudi_config from huggingface model hub(https://huggingface.co/Habana), otherwise use default gaudi_config
        if config["general"].get("gaudi_config_name") is not None:
            gaudi_config = GaudiConfig.from_pretrained(
                config["general"].get("gaudi_config_name"),
            )
        else:
            gaudi_config = GaudiConfig()
            gaudi_config.use_fused_adam = True
            gaudi_config.use_fused_clip_norm = True
        training_args = convert_to_training_args(GaudiTrainingArguments, config)
        trainer = GaudiTrainer(
            model=model,
            args=training_args,
            gaudi_config=gaudi_config,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        common.logger.info("train start")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        common.logger.info("train finish")


def get_finetune_config():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    # Print help if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file) as f:
        finetune_config = parse_yaml_raw_as(FinetuneConfig, f)
    return finetune_config.dict()


def main(external_config=None):
    if not external_config:
        config = get_finetune_config()
    else:
        config = external_config

    config["cwd"] = os.getcwd()

    num_training_workers = config["Training"].get("num_training_workers")
    resources_per_worker = config["Training"].get("resources_per_worker")

    if config["Training"].get("accelerate_mode", None) is None:
        config["Training"]["accelerate_mode"] = "DDP"  # will use DDP to accelerate if no method specified

    ccl_worker_count = 1
    device = config["Training"]["device"]
    if device != "cpu":
        ccl_worker_count = num_training_workers

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "OMP_NUM_THREADS": str(resources_per_worker["CPU"]),
                "CCL_ZE_IPC_EXCHANGE": "sockets",
                "CCL_WORKER_COUNT": str(ccl_worker_count),
                "CCL_LOG_LEVEL": "info",
                "WORLD_SIZE": str(num_training_workers),
                "FI_PROVIDER": "tcp",
            }
        }

        if config["General"]["gpt_base_model"] is True:
            runtime_env["pip"] = ["transformers==4.26.0"]

        if device == "gpu":
            num_cpus = resources_per_worker["CPU"] * num_training_workers + 1  # additional 1 for head worker
            ray.init(num_cpus=num_cpus, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)

    common.logger.info(f"ray available resources = {ray.available_resources()}")
    use_gpu = True if device == "gpu" else False
    scaling_config = ScalingConfig(
        num_workers=num_training_workers,
        use_gpu=use_gpu,
        resources_per_worker=resources_per_worker,
        placement_strategy="SPREAD",
    )

    # if try to use Intel GPU, convert device to 'xpu'
    # due to accelerate internal use 'xpu' represent Intel GPU
    if device == "gpu" and is_xpu_available():
        device = "xpu"

    if config.get("torch_config", None) is None:
        backend = None
        if device == "cpu" or device == "xpu" or device == "gpu":
            backend = "ccl"
        elif device == "hpu":
            backend = "hccl"
        torch_config = common.TorchConfig(backend=backend, device=device)
    else:
        customer_torch_config = config.get("torch_config")
        torch_config = common.TorchConfig(**customer_torch_config, device=device)

    if config.get("failure_config", None) is None:
        failure_config = FailureConfig()
    else:
        customer_failure_config = config.get("failure_config")
        failure_config = FailureConfig(**customer_failure_config)

    if config.get("run_config", None) is None:
        run_config = RunConfig(failure_config=failure_config)
    else:
        customer_run_config = config.get("run_config")
        if customer_run_config.get("failure_config", None) is None:
            customer_run_config["failure_config"] = failure_config
        run_config = RunConfig(**customer_run_config)

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        run_config=run_config,
    )
    results = trainer.fit()
    if external_config is not None:
        return results


if __name__ == "__main__":
    main()
