# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import requests
from langchain_huggingface import HuggingFaceEndpoint

from comps import CustomLogger, GeneratedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import DocSumChatCompletionRequest

from .common import *

logger = CustomLogger("llm_docsum_tgi")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OpeaDocSumTgi")
class OpeaDocSumTgi(OpeaDocSum):
    """A specialized OPEA DocSum TGI component derived from OpeaDocSum for interacting with TGI services based on Lanchain HuggingFaceEndpoint API.

    Attributes:
        client (TGI): An instance of the TGI client for text generation.
    """

    def check_health(self) -> bool:
        """Checks the health of the TGI LLM service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """

        try:
            # response = requests.get(f"{self.llm_endpoint}/health")

            # Will remove after TGI gaudi fix health bug
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

    async def invoke(self, input: DocSumChatCompletionRequest):
        """Invokes the TGI LLM service to generate summarization output for the provided input.

        Args:
            input (DocSumChatCompletionRequest): The input text(s).
        """
        server_kwargs = {}
        if self.access_token:
            server_kwargs["headers"] = {"Authorization": f"Bearer {self.access_token}"}

        if input.stream and input.summary_type == "map_reduce":
            logger.info("Map Reduce mode don't support stream=True, set to stream=False")
            input.stream = False
        self.client = HuggingFaceEndpoint(
            endpoint_url=self.llm_endpoint,
            max_new_tokens=input.max_tokens if input.max_tokens else 1024,
            top_k=input.top_k if input.top_k else 10,
            top_p=input.top_p if input.top_p else 0.95,
            typical_p=input.typical_p if input.typical_p else 0.95,
            temperature=input.temperature if input.temperature else 0.01,
            repetition_penalty=input.repetition_penalty if input.repetition_penalty else 1.03,
            streaming=input.stream,
            server_kwargs=server_kwargs,
            task="text-generation",
        )
        result = await self.generate(input, self.client)

        return result
