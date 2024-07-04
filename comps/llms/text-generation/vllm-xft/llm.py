# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import json

from fastapi.responses import StreamingResponse
from langsmith import traceable
from openai import OpenAI

from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice

llm_endpoint = os.getenv("vLLM_LLM_ENDPOINT", "http://localhost:18688")
model = os.getenv("vLLM_LLM_NAME", "xft")
client = OpenAI(
    api_key="EMPTY",
    base_url=llm_endpoint + "/v1",
)


@register_microservice(
    name="opea_service@llm_vllm_xft",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
@traceable(run_type="llm")
def llm_generate(input: LLMParamsDoc):

    # Create an OpenAI client to interact with the API server
    global client

    completion = client.completions.create(
        model=model,
        prompt=input.query,
        best_of=input.best_of,
        echo=input.echo,
        frequency_penalty=input.frequency_penalty,
        logit_bias=input.logit_bias,
        logprobs=input.logprobs,
        max_tokens=input.max_new_tokens,
        n=input.n,
        presence_penalty=input.presence_penalty,
        seed=input.seed,
        stop=input.stop,
        stream=input.streaming,
        suffix=input.suffix,
        temperature=input.temperature,
        top_p=input.top_p,
        user=input.user,
    )

    if input.streaming:

        def stream_generator():
            chat_response = ""
            for c in completion:
                text = c.choices[0].text
                yield f"data: {json.dumps(text)}\n\n"
            print(f"[llm - chat_stream] stream response: {text}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = completion.choices[0].text
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_vllm_xft"].start()
