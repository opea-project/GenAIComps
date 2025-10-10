# Copyright (C) 2025 Zensar Technologies Private Ltd.
# SPDX-License-Identifier: Apache-2.0

import os

import requests
from langchain_huggingface import HuggingFaceEndpoint

from comps import CustomLogger, OpeaComponentRegistry
from comps.cores.proto.api_protocol import ArbPostHearingAssistantChatCompletionRequest

from .common import *

logger = CustomLogger("arb_post_hearing_assistant_tgi")
logflag = os.getenv("LOGFLAG", False)
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://tgi-server:80")


@OpeaComponentRegistry.register("OpeaArbPostHearingAssistantTgi")
class OpeaArbPostHearingAssistantTgi(OpeaArbPostHearingAssistant):
    """A specialized OPEA TGI component derived from OpeaArbPostHearingAssistantTgi for interacting with TGI services based on Lanchain HuggingFaceEndpoint API.

    Attributes:
        client (TGI): An instance of the TGI client for text generation.
    """

    def check_health(self) -> bool:
        """Checks the health of the TGI LLM service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        try:
            url = f"{self.llm_endpoint}/generate"
            data = {"inputs": "What is Deep Learning?", "parameters": {"max_new_tokens": 17}}
            headers = {"Content-Type": "application/json"}
            response = requests.post(url=url, json=data, headers=headers)

            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            logger.error(e)
            logger.error("Health check failed")
            return False

    async def invoke(self, input: ArbPostHearingAssistantChatCompletionRequest):
        """Invokes the TGI LLM service to generate summarization output for the provided input.

        Args:
            input (ArbPostHearingAssistantChatCompletionRequest): The input text(s).
        """
        self.client = HuggingFaceEndpoint(
            endpoint_url=LLM_ENDPOINT,
            max_new_tokens=input.max_tokens if input.max_tokens else 1024,
            top_k=input.top_k if input.top_k else 10,
            top_p=input.top_p if input.top_p else 0.95,
            typical_p=input.typical_p if input.typical_p else 0.95,
            temperature=input.temperature if input.temperature else 0.01,
            repetition_penalty=input.repetition_penalty if input.repetition_penalty else 1.03,
            timeout=input.timeout if input.timeout is not None else 120,
            task="text-generation",
        )
        result = await self.generate(input, self.client)

        return result
