# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import requests

from comps import CustomLogger, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import EmbeddingRequest, EmbeddingResponse, EmbeddingResponseData

logger = CustomLogger("opea_multimodal_embedding_clip")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_CLIP_EMBEDDING")
class OpeaClipEmbedding(OpeaComponent):
    """A specialized embedding component derived from OpeaComponent for CLIP embedding services.

    This class initializes and configures the CLIP embedding service using the vCLIP model.
    It also performs a health check during initialization and logs an error if the check fails.

    Attributes:
        embeddings (vCLIP): An instance of the vCLIP model used for generating embeddings.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.base_url = os.getenv("CLIP_EMBEDDING_ENDPOINT", "http://localhost:6990")

        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaClipEmbedding health check failed.")

    async def invoke(self, input: EmbeddingRequest) -> EmbeddingResponse:
        """Invokes the embedding service to generate embeddings for the provided input.

        Args:
            input (EmbeddingRequest): The input in OpenAI embedding format, including text(s) and optional parameters like model.

        Returns:
            EmbeddingResponse: The response in OpenAI embedding format, including embeddings, model, and usage information.
        """
        json_payload = input.model_dump()
        try:
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/v1/embeddings",
                headers={"Content-Type": "application/json"},
                json=json_payload,
            )
            response.raise_for_status()
            response_json = response.json()

            return EmbeddingResponse(
                data=[EmbeddingResponseData(**item) for item in response_json.get("data", [])],
                model=response_json.get("model", input.model),
                usage=response_json.get("usage", {}),
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to invoke embedding service: {str(e)}")

    def check_health(self) -> bool:
        """Checks if the embedding model is healthy.

        Returns:
            bool: True if the embedding model is initialized, False otherwise.
        """
        try:
            _ = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers={"Content-Type": "application/json"},
                json={"input": "health check"},
            )

            return True
        except requests.RequestException as e:
            return False
