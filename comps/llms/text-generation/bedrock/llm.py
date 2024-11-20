# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Union

import boto3
from fastapi.responses import StreamingResponse
from langchain_aws import BedrockLLM, ChatBedrock

from comps import (
    CustomLogger,
    GeneratedDoc,
    LLMParamsDoc,
    SearchedDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import ChatCompletionRequest

bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
model_kwargs = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
}

logger = CustomLogger("llm_bedrock")
logflag = os.getenv("LOGFLAG", True)


@register_microservice(
    name="opea_service@llm_bedrock",
    service_type=ServiceType.LLM,
    endpoint="/v1/chat/completions",
    host="0.0.0.0",
    port=9000,
)
def llm_generate(input: Union[LLMParamsDoc, ChatCompletionRequest, SearchedDoc]):
    if logflag:
        logger.info(input)
    content = input.messages[0]["content"]
    llm = ChatBedrock(client=bedrock_runtime, model_id=model_id, model_kwargs=model_kwargs, streaming=input.stream)

    if input.stream:

        async def stream_generator():
            chat_response = ""
            async for text in llm.astream(content):
                chat_response += text.content
                chunk_repr = repr(text.content.encode("utf-8"))
                response = chunk_repr[2:-1].replace("\\n", "\n")
                if logflag:
                    logger.info(f"[llm - chat_stream] chunk:{chunk_repr}")

                # Need to yield data structure similar to TGI for sake of UI
                tgi_format_out = {
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": response}, "finish_reason": None}
                    ]
                }
                yield f"data: {json.dumps(tgi_format_out)}\n\n"
            if logflag:
                logger.info(f"[llm - chat_stream] stream response: {chat_response}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = llm.invoke(content)
        if logflag:
            logger.info(response.content)
        return GeneratedDoc(text=response.content, prompt=content)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_bedrock"].start()
