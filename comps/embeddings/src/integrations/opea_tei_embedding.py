# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import aiohttp
from typing import List, Union
import os
import json
from huggingface_hub import AsyncInferenceClient

from comps.cores.mega.utils import get_access_token
from comps import OpeaComponent, CustomLogger, ServiceType
from comps.cores.proto.api_protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = CustomLogger("opea_tei_embedding")
logflag = os.getenv("LOGFLAG", False)


class OpeaTEIEmbedding(OpeaComponent):
    """
    A specialized embedding component derived from OpeaComponent for TEI embedding services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for embedding generation.
        model_name (str): The name of the embedding model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.client = self._initialize_client()
        self.base_url = os.getenv("TEI_EMBEDDING_ENDPOINT", "http://localhost:8080/v1/embeddings")

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        access_token = get_access_token(
            os.getenv("TOKEN_URL"), os.getenv("CLIENTID"), os.getenv("CLIENT_SECRET")
        )
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        return AsyncInferenceClient(
            model=self.base_url,
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            headers=headers,
        )

    async def invoke(self, input: EmbeddingRequest) -> EmbeddingResponse:
        """
        Invokes the embedding service to generate embeddings for the provided input.

        Args:
            input (EmbeddingRequest): The input in OpenAI embedding format, including text(s) and optional parameters like model.

        Returns:
            EmbeddingResponse: The response in OpenAI embedding format, including embeddings, model, and usage information.
        """
        # Parse input according to the EmbeddingRequest format
        if isinstance(input.input, str):
            texts = [input.input.replace("\n", " ")]
        elif isinstance(input.input, list):
            if all(isinstance(item, str) for item in input.input):
                texts = [text.replace("\n", " ") for text in input.input]
            else:
                raise ValueError("Invalid input format: Only string or list of strings are supported.")
        else:
            raise TypeError("Unsupported input type: input must be a string or list of strings.")
        response = await self.client.post(json={"input": texts, "encoding_format": input.encoding_format, "model": input.model, "user": input.user},
                                          task="text-embedding")
        embeddings = json.loads(response.decode())
        return EmbeddingResponse(**embeddings)

    async def check_health(self) -> bool:
        """
        Checks the health of the embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(f"{self.base_url}/health") as response:
                    # If status is 200, the service is considered alive
                    if response.status == 200:
                        return True
        except aiohttp.ClientError as e:
            # Handle connection errors, timeouts, etc.
            print(f"Health check failed: {e}")
        return False
