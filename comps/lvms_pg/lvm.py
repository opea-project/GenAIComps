# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import os
import time

import requests
from predictionguard import PredictionGuard

from comps import (
    LVMDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)


client = PredictionGuard()


@register_microservice(
    name="opea_service@lvm_pg",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=8091,
    input_datatype=LVMDoc,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@lvm_pg"])
async def lvm(request: LVMDoc):
    start = time.time()
    img_b64_str = request.image
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens

    # make a request to the Prediction Guard API using the LlaVa model
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_b64_str
                    }
                }
            ]
        },
    ]
    result = client.chat.completions.create(
        model="llava-1.5-7b-hf",
        messages=messages,
        max_tokens=max_new_tokens
    )

    statistics_dict["opea_service@lvm_pg"].append_latency(time.time() - start, None)
    return TextDoc(text=result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    print("Prediction Guard LVM initialized.")
    opea_microservices["opea_service@lvm_pg"].start()
