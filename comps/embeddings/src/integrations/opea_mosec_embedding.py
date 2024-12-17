# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import aiohttp
import os
from openai import Client

from comps import OpeaComponent, CustomLogger, ServiceType
from comps.cores.proto.api_protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = CustomLogger("opea_mosec_embedding")
logflag = os.getenv("LOGFLAG", False)

DEFAULT_MODEL = "/home/user/bge-large-zh-v1.5/"

class OpeaMosecEmbedding(OpeaComponent):
    """
    A specialized embedding component derived from OpeaComponent for TEI embedding services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for embedding generation.
        model_name (str): The name of the embedding model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.base_url = os.getenv("MOSEC_EMBEDDING_ENDPOINT", "http://127.0.0.1:8080/")
        self.client = Client(api_key="fake", base_url=self.base_url)

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

        embeddings = self.client.embeddings.create(
            model=input.model or os.environ.get("EMB_MODEL", DEFAULT_MODEL),
            input=texts,
        )
        return embeddings

    async def check_health(self) -> bool:
        """
        Checks the health of the embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            async with aiohttp.ClientSession() as client:
                # There is no '/health' endpoint in MOSEC, use '/metrics' as replacement
                async with client.get(f"{self.base_url}/metrics") as response:
                    # If status is 200, the service is considered alive
                    if response.status == 200:
                        return True
        except aiohttp.ClientError as e:
            # Handle connection errors, timeouts, etc.
            print(f"Health check failed: {e}")
        return False
