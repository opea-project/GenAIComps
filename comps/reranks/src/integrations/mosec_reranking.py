# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import time
from typing import Union

import requests
from langchain_core.prompts import ChatPromptTemplate

from comps import OpeaComponent, CustomLogger, ServiceType

logger = CustomLogger("mosec_reranking")
logflag = os.getenv("LOGFLAG", False)


class MosecReranking(OpeaComponent):
    """
    A specialized reranking component derived from OpeaComponent for Mosec reranking services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for reranking.
        model_name (str): The name of the reranking model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANKING.name.lower(), description, config)
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
        return AsyncInferenceClient(
            model=os.getenv("MOSEC_RERANKING_ENDPOINT", "http://localhost:8080"),
            headers=headers,
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