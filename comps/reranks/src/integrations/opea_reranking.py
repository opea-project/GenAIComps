# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Union

from comps import OpeaComponent, CustomLogger, ServiceType
from comps.cores.mega.utils import get_access_token

logger = CustomLogger("opea_reranking")
logflag = os.getenv("LOGFLAG", False)

# Environment variables
TOKEN_URL = os.getenv("TOKEN_URL")
CLIENTID = os.getenv("CLIENTID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")


class OpeaReranking(OpeaComponent):
    """
    A specialized reranking component derived from OpeaComponent for TEI reranking services.

    Attributes:
        client (AsyncInferenceClient): An instance of the async client for reranking.
        model_name (str): The name of the reranking model used.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANKING.name.lower(), description, config)
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        access_token = get_access_token(
            TOKEN_URL, CLIENTID, CLIENT_SECRET
        )
        headers = {"Content-Type": "application/json"}
        if access_token:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
        return AsyncInferenceClient(
            model=os.getenv("TEI_RERANKING_ENDPOINT", "http://localhost:8080"),
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