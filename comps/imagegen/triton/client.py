import sys

import numpy as np
from PIL import Image
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "stability"
shape = [4]

with httpclient.InferenceServerClient("localhost:18000", network_timeout=1000 * 300) as client:
    queries = [
                "A Buffalo Bill celebrating Superbowl LVIII win",
              ]
    input_arr = [np.frombuffer(bytes(q, 'utf8'), dtype = np.uint8) for q in queries]
    max_size = max([a.size for a in input_arr])
    input_arr = [np.pad(a, (0, max_size-a.size)) for a in input_arr]
    input_arr = np.stack(input_arr)

    inputs = [httpclient.InferInput("INPUT0", input_arr.shape, "UINT8")]
    inputs[0].set_data_from_numpy(input_arr, binary_data=True)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs, 
                            timeout=1000 * 300,
    )

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")

    print(output0_data.size)

    outdir = "."
    for i, b in enumerate(output0_data):
        im = Image.fromarray(np.uint8(b))
        im.save(f"{outdir}/image_{i}.png")
