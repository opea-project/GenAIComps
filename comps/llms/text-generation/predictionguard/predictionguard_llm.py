# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# TODO: Add streaming
import os
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictionguard import PredictionGuard
from comps import GeneratedDoc, ServiceType, opea_microservices, register_microservice

# Set up Prediction Guard API key
PG_API_KEY = os.getenv("PREDICTIONGUARD_API_KEY")
client = PredictionGuard()
# Define the input data model
class LLMParamsDoc(BaseModel):
    model: str = "Neural-Chat-7B"
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
    endpoint="/v1/completions",
    host="0.0.0.0",
    port=9000,
)
def llm_generate(input: LLMParamsDoc):
    

    try:
        response = client.completions.create(
            model=input.model,
            prompt=input.query,
            max_tokens=input.max_new_tokens,
            temperature=input.temperature,
            top_p=input.top_p,
            top_k=input.top_k
        )
        response_text = response['choices'][0]['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GeneratedDoc(text=response_text, prompt=input.query)

if __name__ == "__main__":
    opea_microservices["opea_service@llm_predictionguard"].start()

