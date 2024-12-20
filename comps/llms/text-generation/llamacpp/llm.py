# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import openai
from fastapi.responses import StreamingResponse

from comps import (
    CustomLogger, opea_microservices,
    LLMParamsDoc,
    ServiceType,
    register_microservice, )

logger = CustomLogger("llm_llamacpp")
logflag = os.getenv("LOGFLAG", False)
llamacpp_endpoint = os.getenv("LLAMACPP_ENDPOINT", "http://localhost:8080/")

# OPEA microservice wrapper of llama.cpp
# llama.cpp server uses openai API format: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
@register_microservice(
    name="opea_service@llm_llamacpp",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
async def llm_generate(input: LLMParamsDoc):
    if logflag:
        logger.info(input)
        logger.info(llamacpp_endpoint)

    client = openai.OpenAI(
        base_url=llamacpp_endpoint,  # "http://<Your api-server IP>:port"
        api_key="sk-no-key-required"
    )

    # Llama.cpp works with openai API format
    # The openai api doesn't have top_k parameter
    # https://community.openai.com/t/which-openai-gpt-models-if-any-allow-specifying-top-k/777982/2
    chat_completion = client.chat.completions.create(
        model=input.model,
        messages=[{"role": "user", "content": input.query}],
        max_tokens=input.max_tokens,
        temperature=input.temperature,
        top_p=input.top_p,
        frequency_penalty=input.frequency_penalty,
        presence_penalty=input.presence_penalty,
        stream=input.streaming
    )

    if input.streaming:
        def stream_generator():
            for c in chat_completion:
                if logflag:
                    logger.info(c)
                yield f"data: {c.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        if logflag:
            logger.info(chat_completion)
        return chat_completion


if __name__ == "__main__":
    opea_microservices["opea_service@llm_llamacpp"].start()
