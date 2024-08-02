# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os

import requests

ip_address = os.environ.get("ip_address")
endpoint = f"http://{ip_address}:7860/v1/animation"
inputs = {"image": "", "audio": ""}
response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})
print(response.json())
