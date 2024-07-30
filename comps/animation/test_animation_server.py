# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import requests

endpoint = "http://localhost:7860/v1/animation"
inputs = {"audio_base64_byte_str": "", "image_base64_byte_str": ""}
response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})
print(response.json())