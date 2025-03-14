# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identified: Apache-2.0

import os
import time

import requests
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import ChatCompletionRequest

logger = CustomLogger("opea_textgen_ovms")
logflag = os.getenv("LOGFLAG", False)
MODEL_ID = os.getenv("MODEL_ID")


@OpeaComponentRegistry.register("OpeaTextGenOVMS")
class OpeaTextGenOVMS(OpeaComponent):
    """A specialized OPEA TextGen component derived from OpeaComponent for interacting with OpenVINO Model Server services.

    Attributes:
        client (OpenAI): An instance of the OpenAI client for text generation.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LLM.name.lower(), description, config)
        self.base_url = os.getenv("LLM_ENDPOINT", "http://localhost:8080")
        self.client = OpenAI(base_url=self.base_url + "/v3", api_key="unused")
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaTextGenOVMS health check failed.")
        else:
            logger.info("OpeaTextGenOVMS health check success.")

    def check_health(self) -> bool:
        """Checks the health of the LLM OVMS service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/v2/health/ready")
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            # Handle connection errors, timeouts, etc.
            logger.error(f"Health check failed: {e}")

    async def invoke(self, input: ChatCompletionRequest):
        """Invokes the LLM OVMS service to generate output for the provided input.

        Args:
            input (ChatCompletionRequest): The input text(s).
        """
        if isinstance(input.messages, str):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your goal is to provide accurate, detailed, and safe responses to the user's queries.",
                },
                {"role": "user", "content": input.messages},
            ]
        else:
            messages = input.messages

        optional_params = {}
        if input.top_p:
            optional_params["top_p"] = input.top_p
        if input.top_k:
            optional_params["extra_body"] = {"top_k": input.top_k}
        if input.stream:

            async def stream_generator():
                chat_response = ""
                stream = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=input.max_tokens,
                    temperature=input.temperature,
                    stream=True,
                    **optional_params,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        delta_content = chunk.choices[0].delta.content
                        chat_response += delta_content
                        yield f"data: {delta_content}\n\n"
                    else:
                        yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")
        else:
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_ID,
                    messages=messages,
                    max_tokens=input.max_tokens,
                    temperature=input.temperature,
                    **optional_params,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            return response
