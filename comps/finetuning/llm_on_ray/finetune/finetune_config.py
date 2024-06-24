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

from pydantic import BaseModel, validator
from typing import Optional, List


PRECISION_BF16 = "bf16"
PRECISION_FP16 = "fp16"
PRECISION_NO = "no"

DEVICE_CPU = "cpu"
DEVICE_HPU = "hpu"
DEVICE_GPU = "gpu"

ACCELERATE_STRATEGY_DDP = "DDP"
ACCELERATE_STRATEGY_FSDP = "FSDP"
ACCELERATE_STRATEGY_DEEPSPEED = "DEEPSPEED"


class GeneralConfig(BaseModel):
    trust_remote_code: bool
    use_auth_token: Optional[str]


class LoraConfig(BaseModel):
    task_type: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: Optional[List[str]] = None


class DeltatunerConfig(BaseModel):
    algo: str
    denas: bool
    best_model_structure: str


class General(BaseModel):
    base_model: str
    tokenizer_name: Optional[str] = None
    gaudi_config_name: Optional[str] = None
    gpt_base_model: bool
    output_dir: str
    resume_from_checkpoint: Optional[str] = None
    save_strategy: str = "no"
    config: GeneralConfig
    lora_config: Optional[LoraConfig] = None
    deltatuner_config: Optional[DeltatunerConfig] = None
    enable_gradient_checkpointing: bool = False


class Dataset(BaseModel):
    train_file: str
    validation_file: Optional[str]
    validation_split_percentage: int
    max_length: int = 512
    group: bool = True
    block_size: int = 512
    shuffle: bool = False


class RayResourceConfig(BaseModel):
    CPU: int
    GPU: int = 0
    HPU: int = 0


class Training(BaseModel):
    optimizer: str
    batch_size: int
    epochs: int
    max_train_steps: Optional[int] = None
    learning_rate: float
    lr_scheduler: str
    weight_decay: float
    device: str = DEVICE_CPU
    hpu_execution_mode: str = "lazy"
    num_training_workers: int
    resources_per_worker: RayResourceConfig
    accelerate_mode: str = ACCELERATE_STRATEGY_DDP
    mixed_precision: str = PRECISION_NO
    gradient_accumulation_steps: int = 1
    logging_steps: int = 10
    deepspeed_config_file: str = ""

    @validator("device")
    def check_device(cls, v: str):
        # will convert to lower case
        if v:
            assert v.lower() in [DEVICE_CPU, DEVICE_GPU, DEVICE_HPU]
        return v.lower()

    @validator("hpu_execution_mode")
    def check_hpu_execution_mode(cls, v: str):
        if v:
            assert v in ["lazy", "eager", "eager.compile"]
        return v

    @validator("accelerate_mode")
    def check_accelerate_mode(cls, v: str):
        if v:
            assert v in [
                ACCELERATE_STRATEGY_DDP,
                ACCELERATE_STRATEGY_FSDP,
                ACCELERATE_STRATEGY_DEEPSPEED,
            ]
        return v

    @validator("mixed_precision")
    def check_mixed_precision(cls, v: str):
        if v:
            assert v in [PRECISION_BF16, PRECISION_FP16, PRECISION_NO]
        return v

    @validator("logging_steps")
    def check_logging_steps(cls, v: int):
        assert v > 0
        return v

    # @model_validator(mode='after')
    # def check_device_and_accelerate_mode(self) -> "Training":
    #     dev = self.device
    #     res = self.resources_per_worker
    #     mode = self.accelerate_mode
    #     if dev == "CPU":
    #         if res.GPU is not None and res.GPU > 0:
    #             raise ValueError("Please not specified GPU resource when use CPU only in Ray.")
    #         if mode != "CPU_DDP":
    #             raise ValueError("Please specified CPU related accelerate mode when use CPU only in Ray.")
    #     elif dev == "GPU":
    #         if res.GPU is None or res.GPU == 0:
    #             raise ValueError("Please specified GPU resource when use GPU to fine tune in Ray.")
    #         if mode not in ["GPU_DDP", "GPU_FSDP"]:
    #             raise ValueError("Please speicifed GPU related accelerate mode when use GPU to fine tune in Ray.")

    #     return self


class FinetuneConfig(BaseModel):
    General: General
    Dataset: Dataset
    Training: Training
