# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Union

import boto3
from fastapi.responses import StreamingResponse

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
    kwargs = {
        "modelId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": [{"type": "text", "text": content}]}],
            }
        ),
    }

    if input.stream:

        def stream_generator():
            response = bedrock_runtime.invoke_model_with_response_stream(**kwargs)
            for event in response.get("body"):
                chunk = json.loads(event["chunk"]["bytes"])
                print(chunk)
                if chunk["type"] == "content_block_delta":
                    text_chunk = chunk["delta"]["text"]
                    chunk_repr = repr(text_chunk.encode("utf-8"))
                    if logflag:
                        logger.info(f"[llm - chat_stream] chunk:{chunk_repr}")
                    yield f"data: {chunk_repr}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        response = bedrock_runtime.invoke_model(**kwargs)
        response_body = json.loads(response.get("body").read())
        generated_text = response_body.get("content")[0].get("text")
        print(generated_text)
        if logflag:
            logger.info(generated_text)
        return GeneratedDoc(text=generated_text, prompt=content)


if __name__ == "__main__":
    opea_microservices["opea_service@llm_bedrock"].start()
