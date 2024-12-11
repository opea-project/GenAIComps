# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

from typing import List, Union
import os
import json
from huggingface_hub import AsyncInferenceClient

from comps.cores.mega.utils import get_access_token
from comps import OpeaComponent, CustomLogger, ServiceType

logger = CustomLogger("opea_embedding")
logflag = os.getenv("LOGFLAG", False)


class OpeaEmbedding(OpeaComponent):
    """
    A specialized embedding component derived from OpeaComponent for TEI Gaudi embedding services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for embedding generation.
        model_name (str): The name of the embedding model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        access_token = get_access_token(
            os.getenv("TOKEN_URL"), os.getenv("CLIENTID"), os.getenv("CLIENT_SECRET")
        )
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        return AsyncInferenceClient(
            model=os.getenv("TEI_EMBEDDING_ENDPOINT", "http://localhost:8080"),
            token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            headers=headers,
        )

    async def invoke(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Invokes the embedding service to generate embeddings for the provided input.

        Args:
            input (Union[str, List[str]]): The input text(s) for which embeddings are to be generated.

        Returns:
            List[List[float]]: A list of embedding vectors for the input text(s).
        """
        texts = [input] if isinstance(input, str) else input
        texts = [text.replace("\n", " ") for text in texts]
        response = await self.client.post(json={"inputs": texts}, task="text-embedding")
        embeddings = json.loads(response.decode())
        return embeddings

    def check_health(self) -> bool:
        """
        Checks the health of the embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            response = self.client.get("/health")  # Assuming /health endpoint exists
            return response.status_code == 200 and response.json().get("status", "") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False