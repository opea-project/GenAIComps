# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0


import os
from typing import List, Union

from predictionguard import PredictionGuard

from comps import CustomLogger, OpeaComponent, ServiceType

logger = CustomLogger("predictionguard_embedding")
logflag = os.getenv("LOGFLAG", False)


class PredictionguardEmbedding(OpeaComponent):
    """A specialized embedding component derived from OpeaComponent for interacting with Prediction Guard services.

    Attributes:
        client (PredictionGuard): An instance of the PredictionGuard client for embedding generation.
        model_name (str): The name of the embedding model used by the Prediction Guard service.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.EMBEDDING.name.lower(), description, config)
        self.client = PredictionGuard()
        self.model_name = config.get("PG_EMBEDDING_MODEL_NAME", "bridgetower-large-itm-mlm-itc")

    def check_health(self) -> bool:
        """Checks the health of the Prediction Guard embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            # Simulating a health check request to Prediction Guard
            response = self.client.health_check()
            return response.get("status", "") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def invoke(self, input: Union[str, List[str]]) -> List[List[float]]:
        """Invokes the Prediction Guard embedding service to generate embeddings for the provided input.

        Args:
            input (Union[str, List[str]]): The input text(s) for which embeddings are to be generated.

        Returns:
            List[List[float]]: A list of embedding vectors for the input text(s).
        """
        try:
            texts = [input] if isinstance(input, str) else input
            request_payload = [{"text": text} for text in texts]

            response = self.client.embeddings.create(model=self.model_name, input=request_payload)["data"]
            embeddings = [item["embedding"] for item in response]

            logger.info(f"Generated embeddings for input: {input}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
