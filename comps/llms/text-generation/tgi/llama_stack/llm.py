# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from fastapi.responses import StreamingResponse
from llama_stack_client import AsyncLlamaStackClient
from llama_stack_client.types import UserMessage

from comps import (
    CustomLogger,
    GeneratedDoc,
    LLMParamsDoc,
    ServiceType,
    opea_microservices,
    opea_telemetry,
    register_microservice,
)

logger = CustomLogger("llm_tgi_llama_stack")
logflag = os.getenv("LOGFLAG", False)

@register_microservice(
    name="opea_service@llm_tgi_llama_stack",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
async def llm_generate(input: LLMParamsDoc):
    if logflag:
        logger.info(input)
    llm_endpoint = os.getenv("LLAMA_STACK_ENDPOINT", "http://localhost:5000")
    model_name = os.getenv("LLM_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct").split("/")[-1]

    client = AsyncLlamaStackClient(
        base_url=llm_endpoint,
    )

    response = await client.inference.chat_completion(
        messages=[
            UserMessage(
                content=input.query,
                role="user",
            ),
        ],
        model=model_name,
        sampling_params={
            "max_tokens": input.max_tokens,
        },
        stream=input.streaming,
    )

    if input.streaming:

        async def stream_generator():
            async for text in response:
                output = text.event.delta
                yield f"data: {output}\n\n"
            if logflag:
                logger.info(f"[llm - chat_stream] stream response: {output}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = response.completion_message.content
        if logflag:
            logger.info(response)
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_tgi_llama_stack"].start()