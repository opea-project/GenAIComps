# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Union

from fastrag.rankers import IPEXBiEncoderSimilarityRanker
from haystack import Document
from comps import OpeaComponent, CustomLogger, ServiceType

RANKER_MODEL = os.getenv("EMBED_MODEL", "Intel/bge-small-en-v1.5-rag-int8-static")

logger = CustomLogger("fastrag_reranking")
logflag = os.getenv("LOGFLAG", False)

class FastragReranking(OpeaComponent):
    """
    A specialized reranking component derived from OpeaComponent for fastRAG reranking services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for reranking.
        model_name (str): The name of the reranking model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANKING.name.lower(), description, config)
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        
        return AsyncInferenceClient(
            model = IPEXBiEncoderSimilarityRanker(RANKER_MODEL)
            model.warm_up()
        )

    async def invoke(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Invokes the reranking service to reorder the retrieved docs.
        """

        return reranking_results

    def check_health(self) -> bool:
        """
        Checks the health of the reranking service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            response = self.client.get("/health")  # Assuming /health endpoint exists
            return response.status_code == 200 and response.json().get("status", "") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False