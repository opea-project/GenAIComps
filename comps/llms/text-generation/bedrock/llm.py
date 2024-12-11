# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
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

logger = CustomLogger("llm_bedrock")
logflag = os.getenv("LOGFLAG", True)

region = os.getenv("BEDROCK_REGION", "us-west-2")
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region)

model_kwargs = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
}

sse_headers = {"x-accel-buffering": "no", "cache-control": "no-cache", "content-type": "text/event-stream"}


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

    # Parse out arguments for Bedrock converse API
    model_id = input.model if input.model else model
    if logflag:
        logger.info(f"[llm - chat] Using model {model_id}")

    bedrock_args = {"modelId": model_id}

    inference_config = {}
    if input.max_tokens:
        inference_config["maxTokens"] = input.max_tokens

    if input.stop:
        inference_config["stopSequences"] = input.stop

    if input.temperature:
        inference_config["temperature"] = input.temperature

    if input.top_p:
        inference_config["topP"] = input.top_p

    if len(inference_config) > 0:
        bedrock_args["inferenceConfig"] = inference_config

    if logflag and len(inference_config) > 0:
        logger.info(f"[llm - chat] inference_config: {inference_config}")

    # Parse messages from HuggingFace TGI format to bedrock messages format
    # tgi: [{role: "system" | "user", content: "text"}]
    # bedrock: [role: "assistant" | "user", content: {text: "content"}]
    messages = [
        {"role": "assistant" if i.get("role") == "system" else "user", "content": [{"text": i.get("content", "")}]}
        for i in input.messages
    ]

    # Bedrock requires that conversations start with a user prompt
    # TGI allows the first message to be an assistant prompt, defining assistant behavior
    # If the message list starts with an assistant prompt, move that message to the bedrock system prompt
    if len(messages) > 0 and messages[0]["role"] == "assistant":
        system_prompt = messages[0]["content"][0]["text"]
        bedrock_args["system"] = [{"text": system_prompt}]
        messages.pop(0)

    bedrock_args["messages"] = messages

    if logflag:
        logger.info(f"[llm - chat] Bedrock args: {bedrock_args}")

    if input.stream:
        response = bedrock_runtime.converse_stream(**bedrock_args)

        def stream_generator():
            chat_response = ""
            for chunk in response["stream"]:
                if "contentBlockDelta" in chunk:
                    text = chunk.get("contentBlockDelta", {}).get("delta", {}).get("text", "")
                    if logflag:
                        logger.info(f"[llm - chat_stream] chunk:{text}")

                    tgi_format_out = {
                        "object": "chat.completion.chunk",
                        "model": model_id,
                        "created": int(time.time()),
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(tgi_format_out)}\n\n"
            if logflag:
                logger.info(f"[llm - chat_stream] stream response: {chat_response}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), headers=sse_headers)

    response = bedrock_runtime.converse(**bedrock_args)
    output_content = response.get("output", {}).get("message", {}).get("content", [])
    output_text = output_content[0].get("text", "") if len(output_content) > 0 else ""
    prompt = messages[-1].get("content", [{"text": ""}])[0].get("text", "")

    return GeneratedDoc(text=output_text, prompt=prompt)


if __name__ == "__main__":
    model = os.getenv("MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")
    opea_microservices["opea_service@llm_bedrock"].start()
