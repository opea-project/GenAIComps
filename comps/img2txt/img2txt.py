# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import time
import requests
import json, os

from comps import Img2TxtDoc, TextDoc, ServiceType, opea_microservices, register_microservice, register_statistics, statistics_dict


@register_microservice(
    name="opea_service@img2txt",
    service_type=ServiceType.Img2txt,
    endpoint="/v1/img2txt",
    host="0.0.0.0",
    port=9399,
    input_datatype=Img2TxtDoc,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@img2txt"])
async def img2txt(request: Img2TxtDoc):
    start = time.time()
    img_b64_str = request.image
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens

    inputs = {"img_b64_str": img_b64_str, "prompt": prompt, "max_new_tokens": max_new_tokens}

    # forward to the LLaVA server
    response = requests.post(url=f"{img2txt_endpoint}/generate", data=json.dumps(inputs), proxies={"http": None})

    statistics_dict["opea_service@img2txt"].append_latency(time.time() - start, None)
    return TextDoc(text=response.json()["text"])

if __name__ == "__main__":
    img2txt_endpoint = os.getenv("IMG2TXT_ENDPOINT", "http://localhost:8399")

    print("[img2txt] img2txt initialized.")
    opea_microservices["opea_service@img2txt"].start()