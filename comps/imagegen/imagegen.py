# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import sys
import time
from io import BytesIO

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import *

from comps import Base64ByteStrDoc, ServiceType, TextDoc, opea_microservices, opea_telemetry, register_microservice


@opea_telemetry
def generate_image(*, text, triton_endpoint):
    start = time.time()

    network_timeout=1000 * 300
    with httpclient.InferenceServerClient(triton_endpoint, network_timeout=network_timeout) as client:
        queries = [text]
        input_arr = [np.frombuffer(bytes(q, "utf8"), dtype=np.uint8) for q in queries]
        max_size = max([a.size for a in input_arr])
        input_arr = [np.pad(a, (0, max_size - a.size)) for a in input_arr]
        input_arr = np.stack(input_arr)

        inputs = [httpclient.InferInput("INPUT0", input_arr.shape, "UINT8")]
        inputs[0].set_data_from_numpy(input_arr, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput("OUTPUT0"),
        ]

        ## TODO acwrenn
        ## Parameterize for other ImageGen models?
        model_name = "stability"
        response = client.infer(
            model_name,
            inputs,
            request_id=str(1),
            outputs=outputs,
            timeout=network_timeout,
        )

        result = response.get_response()

    output0_data = response.as_numpy("OUTPUT0")
    if len(output0_data) == 0:
        raise Exception("error fetching images from triton server")
    print(f"generated image in {time.time() - start} seconds")
    return output0_data[0].asbytes()


@register_microservice(
    name="opea_service@imagegen",
    service_type=ServiceType.IMAGEGEN,
    endpoint="/v1/images/generate",
    host="0.0.0.0",
    port=9765,
    input_datatype=TextDoc,
    output_datatype=Base64ByteStrDoc,
)
@opea_telemetry
async def generate_image(input: TextDoc):
    triton_endpoint = os.getenv("IMAGE_GEN_TRITON_ENDPOINT", "http://localhost:8080")
    text = input.text
    image = generate_image(text=text, triton_endpoint=triton_endpoint)
    buffered = BytesIO()
    buffered.write(image.tobytes())
    return Base64ByteStrDoc(byte_str=base64.b64encode(buffered.getvalue()))


if __name__ == "__main__":
    print("[imagegen - router] ImageGen initialized.")
    opea_microservices["opea_service@imagegen"].start()
