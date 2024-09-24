import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
import os
import json
from pathlib import Path
import habana_frameworks.torch as htorch
import time


def get_repo_root(model_name_or_path):
    if os.path.exists(model_name_or_path):
        # local path
        return model_name_or_path
    # checks if online or not
    if is_offline_mode():
        print_rank0("Offline mode: forcing local_files_only=True")
    # download only on first process
    allow_patterns = ["*.bin", "*.model", "*.json", "*.txt", "*.py", "*LICENSE"]
    if local_rank == 0:
        snapshot_download(
            model_name_or_path,
            local_files_only=is_offline_mode(),
            cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
            allow_patterns=allow_patterns,
            # ignore_patterns=["*.safetensors"],
        )

    dist.barrier()

    return snapshot_download(
        model_name_or_path,
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
        allow_patterns=allow_patterns,
        # ignore_patterns=["*.safetensors"],
    )


def get_checkpoint_files(model_name_or_path):
    cached_repo_dir = get_repo_root(model_name_or_path)

    # extensions: .bin | .pt
    # creates a list of paths from all downloaded files in cache dir
    file_list = [str(entry) for entry in Path(cached_repo_dir).rglob("*.[sbp][ait][fn][e][t][e][n][s][o][r][s]") if entry.is_file()]
    print(file_list)
    return file_list


def write_checkpoints_json():
    checkpoint_files = get_checkpoint_files(model_id)
    if local_rank == 0:
        # model.config.model_type.upper()
        data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        json.dump(data, open(checkpoints_json, "w"))

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

def print_rank0(*msg):
    if local_rank != 0:
        return
    print(*msg)

#model_id = "/workspace/models/llama3.2/final/Llama-3.2-11B-Vision"
model_id = "/workspace/models/final_weights/Llama-3.2-90B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

deepspeed.init_distributed(dist_backend="hccl")
local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], "1")

repo_root = get_repo_root(model_id)
checkpoints_json = "checkpoints.json"
write_checkpoints_json()

print("sleep for 10 seconds to avoid multi-process conflict")
time.sleep(10)
model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    base_dir=repo_root,
    dtype=torch.bfloat16,
    checkpoint=checkpoints_json,
)
print(model)


print("=================================")

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
url = "https://llava-vl.github.io/static/images/view.jpg"
raw_image = Image.open(requests.get(url, stream=True).raw)

import time
total_time = 0.0
warmup = 5
iters = 10
for i in range(iters):
    t0 = time.time()
    inputs = processor(prompt, raw_image, return_tensors="pt").to(torch.device("hpu"))
    output = model.generate(**inputs, do_sample=False, max_new_tokens=25, pad_token_id=processor.tokenizer.pad_token_id)
    t1 = time.time()
    print_rank0(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
    print_rank0("Iteration: %d, Time: %.6f sec" % (i, t1 - t0))
    if i >= warmup:
        total_time += t1 - t0
latency = total_time / (iters - warmup) * 1000
print_rank0("\n", "-" * 10, "Summary:", "-" * 10)
print_rank0("Inference latency: %.2f ms." % latency)

