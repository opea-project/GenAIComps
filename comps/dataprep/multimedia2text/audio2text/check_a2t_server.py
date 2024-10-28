# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import requests

# Get the root folder of the current script
root_folder = os.path.dirname(os.path.abspath(__file__))

# Path to the test audio file
file_name = os.path.join(root_folder, '../data/test_audio_30s.wav')

# Read and encode the audio file in base64
with open(file_name, "rb") as f:
    test_audio_base64_str = base64.b64encode(f.read()).decode("utf-8")

# Define the endpoint and the input data
endpoint = "http://localhost:9099/v1/audio/transcriptions"
inputs = {"byte_str": test_audio_base64_str}

# Send the POST request to the endpoint
response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})

# Print the response from the server
print(response.json())