# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import habana_frameworks.torch as htorch
from io import BytesIO
import os
from PIL import Image
import requests
import threading
import time
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from typing import Union
from prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion
from comps import (
    CustomLogger,
    LVMDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from enum import Enum

logger = CustomLogger("lvm-llama-vision-guard-native")
logflag = os.getenv("LOGFLAG", False)
initialization_lock = threading.Lock()
initialized = False

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

def initialize():
    global model, processor, initialized
    with initialization_lock:
        if not initialized:
            import habana_frameworks.torch.hpu as torch_hpu
            model_id = os.getenv("LLAMA_VISION_GUARD_MODEL_ID", "meta-llama/Llama-Guard-3-11B-Vision")
            huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="hpu", torch_dtype=torch.bfloat16, token=huggingface_token)
            processor = AutoProcessor.from_pretrained(model_id, token=huggingface_token)
            url = "https://llava-vl.github.io/static/images/view.jpg"
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "If I had to write a haiku for this one, it would be: "
                        },
                        {
                            "type": "image",
                        },
                    ],
                }
            ]
            input_prompt = processor.apply_chat_template(
                conversation, return_tensors="pt"
            )
            raw_image = Image.open(requests.get(url, stream=True).raw)
            inputs = processor(text=input_prompt, images=raw_image, return_tensors="pt").to(model.device)
            prompt_len = len(inputs['input_ids'][0])
            output = model.generate(**inputs, pad_token_id=0, max_new_tokens=32)
            generated_tokens = output[:, prompt_len:]
            logger.info(processor.decode(generated_tokens[0]))
            initialized = True
            logger.info("[LVM] Llama Vision GUARD LVM initialized.")


@register_microservice(
    name="opea_service@lvm_llama_vision_guard_native",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9499,
)
@register_statistics(names=["opea_service@lvm_llama_vision_guard_native"])
async def lvm(request: Union[LVMDoc]) -> Union[TextDoc]:
    initialize()
    if logflag:
        logger.info(request)
    start = time.time()
    img_b64_str = request.image
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens

    llama_guard_version = "LLAMA_GUARD_3"
    prompts = [(prompt, AgentType.USER)]
    for prompt in prompts:
        formatted_prompt = build_default_prompt(
                prompt[1],
                create_conversation([prompt[0]]),
                llama_guard_version)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": formatted_prompt
                },
                {
                    "type": "image",
                },
            ],
        }
    ]
    input_prompt = processor.apply_chat_template(
        conversation, return_tensors="pt"
    )

    image_data = base64.b64decode(img_b64_str)
    image_stream = BytesIO(image_data)
    raw_image = Image.open(image_stream)

    inputs = processor(text=input_prompt, images=raw_image, return_tensors="pt").to(model.device)
    prompt_len = len(inputs['input_ids'][0])
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated_tokens = output[:, prompt_len:]

    statistics_dict["opea_service@lvm_llama_vision_guard_native"].append_latency(time.time() - start, None)
    result = processor.decode(generated_tokens[0])
    if logflag:
        logger.info(result)

    return TextDoc(text=result)


if __name__ == "__main__":
    opea_microservices["opea_service@lvm_llama_vision_guard_native"].start()
