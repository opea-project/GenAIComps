# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictionguard import PredictionGuard
from comps import GeneratedDoc, ServiceType, opea_microservices, register_microservice

client = PredictionGuard()


class LLMParamsDoc(BaseModel):
    model: str = "Neural-Chat-7B"
    query: str
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stream: bool = False


app = FastAPI()


@register_microservice(
    name="opea_service@llm_predictionguard",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)

def llm_generate(input: LLMParamsDoc):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides clever and sometimes funny responses."
        },
        {
            "role": "user",
            "content": input.query
        }
    ]

    if input.stream:
        async def stream_generator():
            try:
                for res in client.chat.completions.create(
                    model=input.model,
                    messages=messages,
                    max_tokens=input.max_new_tokens,
                    temperature=input.temperature,
                    top_p=input.top_p,
                    top_k=input.top_k,
                    stream=True
                ):
                    if 'choices' in res['data'] and 'delta' in res['data']['choices'][0]:
                        delta_content = res['data']['choices'][0]['delta']['content']
                        yield f"{delta_content}"
                    else:
                        yield f"Unexpected response format: {res}\n\n"
            except Exception as e:
                yield f"{str(e)}"
            # finally:
            #     yield "[DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            response = client.chat.completions.create(
                model=input.model,
                messages=messages,
                max_tokens=input.max_new_tokens,
                temperature=input.temperature,
                top_p=input.top_p,
                top_k=input.top_k
            )
            response_text = response['choices'][0]['message']['content']
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return GeneratedDoc(text=response_text, prompt=input.query)



if __name__ == "__main__":
    opea_microservices["opea_service@llm_predictionguard"].start()
#docker run -d -p 9000:9000 -e PREDICTIONGUARD_API_KEY="dzyWIYuiKy4pWjJIX1oab9eZk5zd7T7o212GtM0f" --name predictionguard-llm-container predictionguard-llm
#docker stop $(docker ps -q) && docker rm $(docker ps -a -q)
# docker build -t predictionguard-llm -f comps/llms/text-generation/predictionguard/Dockerfile .