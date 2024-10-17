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

file_name = 'comps/dataprep/docsum/data/test_video_60s.wav'
# # file_name = 'comps/dataprep/docsum/data/test_full.wav'

with open(file_name, "rb") as f:
    test_audio_base64_str = base64.b64encode(f.read()).decode("utf-8")

# # os.remove(file_name)


endpoint = "http://localhost:7079/v1/docsum/dataprep"
# endpoint = "http://localhost:3001/v1/audio/transcriptions"

# text_cont = "and a number of states are starting to adopt them voluntarily special correspondent john delenco of education week reports it takes just 10 minutes to cross through gillette wyoming this small city sits in the northeast corner of the state surrounded by 100s of miles of prairie but schools here in campbell county are on the edge of something big the next generation science standards you are going to build a strand of dna and you are going to decode it and figure out what that dna actually says for christy mathis at sage valley junior high school the new standards are about learning to think like a scientist there is a lot of really good stuff in them every standard is a performance task it is not you know the child needs to memorize these things it is the student needs to be able to do some pretty intense stuff we are analyzing we are critiquing we are ."
    
# # Define the endpoint and payload
# inputs = {"text": text_cont}

#inputs = {"byte_str": "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"} # , "max_tokens":64}

inputs = {"text": " THIS IS A TEST >>>> and a number of states are starting to adopt them voluntarily special correspondent john delenco of education week reports it takes just 10 minutes to cross through gillette wyoming this small city sits in the northeast corner of the state surrounded by 100s of miles of prairie but schools here in campbell county are on the edge of something big the next generation science standards you are going to build a strand of dna and you are going to decode it and figure out what that dna actually says for christy mathis at sage valley junior high school the new standards are about learning to think like a scientist there is a lot of really good stuff in them every standard is a performance task it is not you know the child needs to memorize these things it is the student needs to be able to do some pretty intense stuff we are analyzing we are critiquing we are ."} 
# inputs = {"audio": "UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA"} 
# inputs = {"audio": test_audio_base64_str} 

response = requests.post(url=endpoint, data=json.dumps(inputs), proxies={"http": None})
print(response.json())
