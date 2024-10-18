# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
import urllib.request
import uuid
from io import BytesIO

import requests

# https://gist.github.com/novwhisky/8a1a0168b94f3b6abfaa
# test_audio_base64_str = "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"

# uid = str(uuid.uuid4())
# file_name = uid + ".wav"

# urllib.request.urlretrieve(
#     "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav",
#     file_name,
# )

endpoint = "http://localhost:7079/v1/docsum/dataprep"

def get_base64_str(file_name):
    with open(file_name, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# file_name = 'comps/dataprep/docsum/data/test_full.mp4'
file_name = 'comps/dataprep/docsum/data/test_video_30s.mp4'

# file_name = 'comps/dataprep/docsum/data/test_full.wav'
# file_name = 'comps/dataprep/docsum/data/test_video_30s.wav'

test_audio_base64_str = get_base64_str(file_name)

# inputs = {"text": " THIS IS A TEST >>>> and a number of states are starting to adopt them voluntarily special correspondent john delenco of education week reports it takes just 10 minutes to cross through gillette wyoming this small city sits in the northeast corner of the state surrounded by 100s of miles of prairie but schools here in campbell county are on the edge of something big the next generation science standards you are going to build a strand of dna and you are going to decode it and figure out what that dna actually says for christy mathis at sage valley junior high school the new standards are about learning to think like a scientist there is a lot of really good stuff in them every standard is a performance task it is not you know the child needs to memorize these things it is the student needs to be able to do some pretty intense stuff we are analyzing we are critiquing we are ."} 

# inputs = {"audio": test_audio_base64_str} 
inputs = {"video": test_audio_base64_str} 

response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})
print(response.json())
