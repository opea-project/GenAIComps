# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

import json
import os
import time

import boto3
from botocore.exceptions import ClientError
from fastapi.responses import StreamingResponse

from comps import CustomLogger, GeneratedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import ChatCompletionRequest

logger = CustomLogger("opea_textgen_bedrock")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OpeaTextGenBedrock")
class OpeaTextGenBedrock(OpeaComponent):
    """A specialized OPEA TextGen component derived from OpeaComponent for
    interacting with AWS Bedrock."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LLM.name.lower(), description, config)

        self.region = os.getenv("BEDROCK_REGION", "us-west-2")
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=self.region)
        self.sts_client = boto3.client("sts", region_name=self.region)

        self.sse_headers = {"x-accel-buffering": "no", "cache-control": "no-cache", "content-type": "text/event-stream"}

        self.default_model = os.getenv("MODEL_ID", "us.anthropic.claude-3-haiku-20240307-v1:0")

        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaTextGenBedrock health check failed.")

    def check_health(self):
        """Checks health by validating ability to check caller identity with
        AWS.

        Returns:
            bool: True if AWS is reachable, False otherwise
        """
        try:
            response = self.sts_client.get_caller_identity()
            return response is not None
        except ClientError as e:
            logger.error(e)
            logger.error("OpeaTextGenBedrock health check failed")
            return False

    async def invoke(self, input: ChatCompletionRequest):
        """Invokes the AWS Bedrock service to generate a response based on the
        previous chats.

        Args:
            input (ChatCompletionRequest): The chat input.
        """
        if logflag:
            logger.info(input)

        # Parse out arguments for Bedrock converse API
        model_id = input.model if input.model else self.default_model
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

        # Parse messages to Bedrock format
        # tgi: "prompt" or [{role: "system" | "user", content: "text"}]
        # bedrock: [role: "assistant" | "user", content: {text: "content"}]
        messages = None
        if isinstance(input.messages, str):
            messages = [{"role": "user", "content": [{"text": input.messages}]}]
        else:
            # Convert from list of HuggingFace TGI message objects
            messages = [
                {
                    "role": "assistant" if i.get("role") == "system" else "user",
                    "content": [{"text": i.get("content", "")}],
                }
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
            response = self.bedrock_runtime.converse_stream(**bedrock_args)

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

            return StreamingResponse(stream_generator(), headers=self.sse_headers)

        response = self.bedrock_runtime.converse(**bedrock_args)
        output_content = response.get("output", {}).get("message", {}).get("content", [])
        output_text = output_content[0].get("text", "") if len(output_content) > 0 else ""
        prompt = messages[-1].get("content", [{"text": ""}])[0].get("text", "")

        return GeneratedDoc(text=output_text, prompt=prompt)
