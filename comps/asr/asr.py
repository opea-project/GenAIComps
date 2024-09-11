# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import numpy as np
import requests

from comps import (
    Base64ByteStrDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)


@register_microservice(
    name="opea_service@asr",
    service_type=ServiceType.ASR,
    endpoint="/v1/audio/transcriptions",
    host="0.0.0.0",
    port=9099,
    input_datatype=Base64ByteStrDoc,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@asr"])
async def audio_to_text(audio: Base64ByteStrDoc):
    start = time.time()
    byte_str = audio.byte_str
    inputs = {"audio": byte_str}

    response = requests.post(url=f"{asr_endpoint}/v1/asr", data=json.dumps(inputs), proxies={"http": None})

    statistics_dict["opea_service@asr"].append_latency(time.time() - start, None)
    return TextDoc(text=response.json()["asr_result"])


if __name__ == "__main__":
    asr_endpoint = os.getenv("ASR_ENDPOINT", "http://localhost:7066")
    print("[asr - router] ASR initialized.")
    opea_microservices["opea_service@asr"].start()
