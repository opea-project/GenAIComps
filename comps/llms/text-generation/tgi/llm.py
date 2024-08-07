# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from langsmith import traceable

from comps import (
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
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
async def llm_generate(input: LLMParamsDoc):
    stream_gen_time = []
    start = time.time()
    if input.streaming:

        async def stream_generator():
            chat_response = ""
            text_generation = await llm.text_generation(
                prompt=input.query,
                stream=input.streaming,
                max_new_tokens=input.max_new_tokens,
                repetition_penalty=input.repetition_penalty,
                temperature=input.temperature,
                top_k=input.top_k,
                top_p=input.top_p,
            )
            async for text in text_generation:
                stream_gen_time.append(time.time() - start)
                chat_response += text
                chunk_repr = repr(text.encode("utf-8"))
                print(f"[llm - chat_stream] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            print(f"[llm - chat_stream] stream response: {chat_response}")
            statistics_dict["opea_service@llm_tgi"].append_latency(stream_gen_time[-1], stream_gen_time[0])
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = await llm.text_generation(
            prompt=input.query,
            stream=input.streaming,
            max_new_tokens=input.max_new_tokens,
            repetition_penalty=input.repetition_penalty,
            temperature=input.temperature,
            top_k=input.top_k,
            top_p=input.top_p,
        )
        statistics_dict["opea_service@llm_tgi"].append_latency(time.time() - start, None)
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    llm_endpoint = os.getenv("TGI_LLM_ENDPOINT", "http://localhost:8080")
    llm = AsyncInferenceClient(
        model=llm_endpoint,
        timeout=600,
    )
    opea_microservices["opea_service@llm_tgi"].start()
