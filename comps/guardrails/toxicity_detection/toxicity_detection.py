# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import sys
from datetime import datetime

cur_path = pathlib.Path(__file__).parent.resolve()
comps_path = os.path.join(cur_path, "../../../")
sys.path.append(comps_path)

import torch
from fastapi.responses import StreamingResponse
from langsmith import traceable

# from utils import initialize_model
from transformers import pipeline

from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice


@register_microservice(
    name="opea_service@toxicity_detection",
    service_type=ServiceType.LLM,
    endpoint="/v1/toxicity",
    host="0.0.0.0",
    port=9091,
)
@traceable(run_type="llm")
async def llm_generate(input: LLMParamsDoc):
    input_query = input.query
    model_name_or_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    toxicity_classifier = pipeline("text-classification", model=model_name_or_path, tokenizer=model_name_or_path)
    toxic = toxicity_classifier(input_query)
    if toxic[0]["label"] == "toxic":
        return f"\nI'm sorry, but your query or LLM's response is TOXIC with an score of {toxic[0]['score']:.2f} (0-1)!!!\n"
    else:
        return input_query


if __name__ == "__main__":
    opea_microservices["opea_service@toxicity_detection"].start()
