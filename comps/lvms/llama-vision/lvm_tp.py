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
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Union

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

logger = CustomLogger("lvm-llama-vision-tp-native")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@lvm_llama_vision_tp_native",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9599,
)
@register_statistics(names=["opea_service@lvm_llama_vision_tp_native"])
async def lvm(request: Union[LVMDoc]) -> Union[TextDoc]:
    if logflag:
        logger.info(request)
    start = time.time()
    img_b64_str = request.image
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens

    text = f"<|image|><|begin_of_text|>{prompt}"

    image_data = base64.b64decode(img_b64_str)
    image_stream = BytesIO(image_data)
    raw_image = Image.open(image_stream)


    return TextDoc(text=result)


if __name__ == "__main__":
    opea_microservices["opea_service@lvm_llama_vision_tp_native"].start()

