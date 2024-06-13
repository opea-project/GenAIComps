# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi.responses import StreamingResponse
from langsmith import traceable
from openai import OpenAI

from comps import GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice


@register_microservice(
    name="opea_service@llm_vllm_xft",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
@traceable(run_type="llm")
def llm_generate(input: LLMParamsDoc):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:18688/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    if input.streaming:

        def stream_generator():
            chat_response = ""
            response = client.completions.create(
                model="xft",
                prompt=input.query,
                max_tokens=input.max_new_tokens,
                temperature=input.temperature,
                top_p=input.top_p,
                n=input.top_k,
                frequency_penalty=input.repetition_penalty,
                stream=input.streaming,
            )
            for chunk in response:
                chat_response += chunk.choices[0].delta.content
                chunk_repr = repr(chunk.choices[0].delta.content.encode("utf-8"))
                print(f"[llm - chat_stream] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            print(f"[llm - chat_stream] stream response: {chat_response}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = client.completions.create(
            model="xft",
            prompt=input.query,
            max_tokens=input.max_new_tokens,
            temperature=input.temperature,
            top_p=input.top_p,
            n=input.top_k,
            frequency_penalty=input.repetition_penalty,
        )
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_vllm_xft"].start()
