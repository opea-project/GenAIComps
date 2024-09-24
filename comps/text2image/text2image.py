# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import os
import time

import requests

from comps import (
    SDInputs,
    SDOutputs,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)


@register_microservice(
    name="opea_service@text2image",
    service_type=ServiceType.TEXT2IMAGE,
    endpoint="/v1/text2image",
    host="0.0.0.0",
    port=9379,
    input_datatype=SDInputs,
    output_datatype=SDOutputs,
)
@register_statistics(names=["opea_service@text2image"])
async def text2image(input: SDInputs):
    start = time.time()
    inputs = {"prompt": input.prompt, "num_images_per_prompt": input.num_images_per_prompt}
    images = requests.post(url=f"{sd_endpoint}/generate", data=json.dumps(inputs), proxies={"http": None}).json()[
        "images"
    ]

    statistics_dict["opea_service@text2image"].append_latency(time.time() - start, None)
    return SDOutputs(images=images)


if __name__ == "__main__":
    sd_endpoint = os.getenv("SD_ENDPOINT", "http://localhost:9378")
    print("Text2image server started.")
    opea_microservices["opea_service@text2image"].start()
