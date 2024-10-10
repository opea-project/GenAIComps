# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from fastapi.responses import StreamingResponse
from langchain_community.llms import Ollama

from comps import CustomLogger, GeneratedDoc, LLMParamsDoc, ServiceType, opea_microservices, register_microservice

logger = CustomLogger("llm_ollama")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@llm_ollama",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
async def llm_generate(input: LLMParamsDoc):
    if logflag:
        logger.info(input)
    ollama = Ollama(
        base_url=ollama_endpoint,
        model=input.model if input.model else model_name,
        num_predict=input.max_tokens,
        top_k=input.top_k,
        top_p=input.top_p,
        temperature=input.temperature,
        repeat_penalty=input.repetition_penalty,
    )
    # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3`
    if input.streaming:

        async def stream_generator():
            chat_response = ""
            async for text in ollama.astream(input.query):
                chat_response += text
                chunk_repr = repr(text.encode("utf-8"))
                if logflag:
                    logger.info(f"[llm - chat_stream] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            if logflag:
                logger.info(f"[llm - chat_stream] stream response: {chat_response}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = await ollama.ainvoke(input.query)
        if logflag:
            logger.info(response)
        return GeneratedDoc(text=response, prompt=input.query)


if __name__ == "__main__":
    ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    opea_microservices["opea_service@llm_ollama"].start()
