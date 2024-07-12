# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictionguard import PredictionGuard
from comps import GeneratedDoc, ServiceType, opea_microservices, register_microservice

# Set up Prediction Guard API key
PG_API_KEY = os.getenv("PREDICTIONGUARD_API_KEY")

# Define the input data model
class LLMParamsDoc(BaseModel):
    query: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    #stream: bool = False

app = FastAPI()

@register_microservice(
    name="opea_service@llm_predictionguard",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
def llm_generate(input: LLMParamsDoc):
    if not PG_API_KEY:
        raise HTTPException(status_code=500, detail="PREDICTION_GUARD_API_KEY environment variable is not set")

    client = PredictionGuard()

    try:
        response = client.completions.create(
            model="Neural-Chat-7B",
            prompt=input.query,
            max_tokens=input.max_new_tokens,
            temperature=input.temperature,
            top_p=input.top_p,
            top_k=input.top_k
            #stream=input.stream
        )
        response_text = response['choices'][0]['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GeneratedDoc(text=response_text, prompt=input.query)

if __name__ == "__main__":
    # This is needed to set up and start the microservice.
    opea_microservices["opea_service@llm_predictionguard"].start()

# # Copyright (C) 2024 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

# import os
# from fastapi.responses import StreamingResponse
# from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice
# from predictionguard import PredictionGuard
# import os
# import requests

# breakpoint()
# # Initialize Prediction Guard API URL and Key
# PG_API_URL = "https://api.predictionguard.com/completions"
# PG_API_KEY = os.getenv("PREDICTION_GUARD_API_KEY")

# @register_microservice(
#     name="opea_service@llm_predictionguard",
#     service_type=ServiceType.LLM,
#     endpoint="/v1/chat/completions",
#     host="0.0.0.0",
#     port=9000,
# )

# def llm_generate(input: LLMParamsDoc):
#     headers = {
#         "Authorization": f"Bearer {PG_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": "Neural-Chat-7B",
#         "prompt": input.query,
#         "max_tokens": input.max_new_tokens,
#         "temperature": input.temperature,
#         "top_p": input.top_p,
#         "top_k": input.top_k,
#         "stream": input.stream
#     }
#     # TODO: figure out how this works with streaming
#     response = requests.post(PG_API_URL, headers=headers, json=data)
#     response.raise_for_status()
#     result = response.json()
#     response_text = result['choices'][0]['text']
#     return response_text

# if __name__ == "__main__":
#     opea_microservices["opea_service@llm_predictionguard"].start()
