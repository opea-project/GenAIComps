# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from langsmith import traceable

from comps import (
    GeneratedDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import (
        LLMChatCompletionRequest,
        ChatCompletionStreamResponse,
        ChatCompletionResponse,
)
from openai import OpenAI
import json

llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")

client = OpenAI(
    api_key="EMPTY",
    base_url=llm_endpoint + "/v1",
)


@register_microservice(
    name="opea_service@llm_tgi",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
@traceable(run_type="llm")
@register_statistics(names=["opea_service@llm_tgi"])
async def llm_generate(request: LLMChatCompletionRequest):
    stream_gen_time = []
    start = time.time()

    chat_completion = client.chat.completions.create(
            model="tgi",
            messages=request.messages,
            frequency_penalty=request.frequency_penalty,
            logit_bias=request.logit_bias,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
            max_tokens=request.max_tokens,
            n=request.n,
            presence_penalty=request.presence_penalty,
            response_format=request.response_format,
            seed=request.seed,
            service_tier=request.service_tier,
            stop=request.stop,
            stream=request.stream,
            stream_options=request.stream_options,
            temperature=request.temperature,
            top_p=request.top_p,
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            user=request.user,
    )

    if request.stream:
        def stream_generator():
            for c in chat_completion:
                text = c.choices[0].delta.content
                print(f"[llm - chat_stream] chunk: {text}")
                yield f"data: {json.dumps(text)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = chat_completion.choices[0].message.content
        return GeneratedDoc(text=response, prompt="")

if __name__ == "__main__":
    opea_microservices["opea_service@llm_tgi"].start()
