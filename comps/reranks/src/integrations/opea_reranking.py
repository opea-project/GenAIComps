# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union

import json
import time

import asyncio

from comps import (
    CustomLogger,
    LLMParamsDoc,
    SearchedDoc,
    ServiceType,
    OpeaComponent,
)

from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    RerankingRequest,
    RerankingResponse,
    RerankingResponseData,
)

from comps.cores.mega.utils import get_access_token
from huggingface_hub import AsyncInferenceClient

logger = CustomLogger("opea_reranking")
logflag = os.getenv("LOGFLAG", False)

# Environment variables
TOKEN_URL = os.getenv("TOKEN_URL", )
CLIENTID = os.getenv("CLIENTID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


class OpeaReranking(OpeaComponent):
    """
    A specialized reranking component derived from OpeaComponent for TEI reranking services.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANK.name.lower(), description, config)
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        access_token = (
            get_access_token(TOKEN_URL, CLIENTID, CLIENT_SECRET) if TOKEN_URL and CLIENTID and CLIENT_SECRET else None
        )
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}

        # Return the initialized AsyncInferenceClient
        return AsyncInferenceClient(
            model=os.getenv("TEI_EMBEDDING_ENDPOINT", "http://localhost:8000"),
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            headers=headers,
        )

    async def invoke(self, input):
        """
        Asynchronously invokes the reranking process with the given input.

        Args:
            input (dict): The input data for the reranking process.

        Returns:
            dict: The response data from the reranking process.

        Comments:
            - Initializes an empty list for reranking results.
            - Retrieves an access token if the necessary credentials are provided.
            - Sets the headers for the HTTP request, including the access token if available.
            - Sends a POST request to the client with the input data and the task "text-reranking".
        """
        if logflag:
            logger.info(input)

        start = time.time()
        reranking_results = []
        access_token = (
            get_access_token(TOKEN_URL, CLIENTID, CLIENT_SECRET) if TOKEN_URL and CLIENTID and CLIENT_SECRET else None
        )

        headers = {"Content-Type": "application/json"}

        if access_token:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}

        # Send a POST request to the client with the input data and the task "text-reranking"
        response_data = await self.client.post(json=input, task="text-reranking")

        return response_data

    def check_health(self) -> bool:
        """
        Checks the health of the reranking service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            # Send a POST request to the client with a sample query and texts to check the health of the service
            response = asyncio.run(self.client.post(json={"query": "hi", "texts": ["Hello", "Fine"]}, task="text-reranking"))
            response = json.loads(response.decode('utf-8'))

            if (response[0]["index"] is not None and response[0]["score"] is not None):
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
