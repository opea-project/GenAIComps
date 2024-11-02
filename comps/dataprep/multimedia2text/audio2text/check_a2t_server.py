# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import requests
import uuid
import urllib.request


uid = str(uuid.uuid4())
file_name = uid + ".wav"

urllib.request.urlretrieve(
    "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav",
    file_name,
)

# Read and encode the audio file in base64
with open(file_name, "rb") as f:
    test_audio_base64_str = base64.b64encode(f.read()).decode("utf-8")
os.remove(file_name)

# Define the endpoint and the input data
endpoint = "http://localhost:9099/v1/audio/transcriptions"
inputs = {"byte_str": test_audio_base64_str}

# Send the POST request to the endpoint
response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})

# Print the response from the server
print(response.json())