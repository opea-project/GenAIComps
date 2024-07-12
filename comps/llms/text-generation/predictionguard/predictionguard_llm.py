# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from fastapi.responses import StreamingResponse
from langsmith import traceable
from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice
from predictionguard import PredictionGuard
import os
import requests
from langsmith import traceable

# Initialize Prediction Guard API URL and Key
PG_API_URL = "https://api.predictionguard.com/completions"
PG_API_KEY = os.getenv("PREDICTION_GUARD_API_KEY")

@register_microservice(
    name="opea_service@llm_predictionguard",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
@traceable(run_type="llm")
def llm_generate(input: LLMParamsDoc):
    headers = {
        "Authorization": f"Bearer {PG_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "Neural-Chat-7B",
        "prompt": input.query,
        "max_tokens": input.max_new_tokens,
        "temperature": input.temperature,
        "top_p": input.top_p,
        "top_k": input.top_k,
        "streaming": input.stream
    }
    # TODO: figure out how this works with streaming
    # if input.streaming:
    #     async def stream_generator():
    #         response = requests.post(PG_API_URL, headers=headers, json=data, stream=True)
    #         chat_response = ""
    #         for line in response.iter_lines():
    #             if line:
    #                 text = line.decode('utf-8')
    #                 chat_response += text
    #                 chunk_repr = repr(text.encode("utf-8"))
    #                 print(f"[llm - chat_stream] chunk:{chunk_repr}")
    #                 yield f"data: {chunk_repr}\n\n"
    #         print(f"[llm - chat_stream] stream response: {chat_response}")
    #         yield "data: [DONE]\n\n"

    #     return StreamingResponse(stream_generator(), media_type="text/event-stream")
    #else:
    response = requests.post(PG_API_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    response_text = result['choices'][0]['text']
    return GeneratedDoc(text=response_text, prompt=input.query)

if __name__ == "__main__":
    opea_microservices["opea_service@llm_predictionguard"].start()
