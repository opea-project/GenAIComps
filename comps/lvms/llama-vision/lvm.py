# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import time
import torch
from typing import Union

from transformers import MllamaForConditionalGeneration, AutoProcessor


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

logger = CustomLogger("lvm-llama-vision-native")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@lvm_llama_vision_native",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9399,
)
@register_statistics(names=["opea_service@lvm_llama_vision_native"])
async def lvm(request: Union[LVMDoc]) -> Union[TextDoc]:
    if logflag:
        logger.info(request)
    start = time.time()
    img_b64_str = request.image
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens

    inputs = processor(prompt, img_b64_str, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    statistics_dict["opea_service@lvm_llama_vision_native"].append_latency(time.time() - start, None)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    if logflag:
        logger.info(result)

    return TextDoc(text=result)


if __name__ == "__main__":
    model_id = os.getenv("LLAMA_VISION_MODEL_ID", "meta-llama/Llama-3.2-11B-Vision-Instruct")
    model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="hpu", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_id)
    logger.info("[LVM] Llama Vision LVM initialized.")
    opea_microservices["opea_service@lvm_llama_vision_native"].start()
