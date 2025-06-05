# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Union

import requests
from huggingface_hub import AsyncInferenceClient

from comps import CustomLogger, LLMParamsDoc, OpeaComponentRegistry, SearchedDoc, ServiceType
from comps.cores.common.component import OpeaComponent
from comps.cores.mega.utils import get_access_token
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    RerankingRequest,
    RerankingResponse,
    RerankingResponseData,
)

logger = CustomLogger("ovms_reranking")
logflag = os.getenv("LOGFLAG", False)

# Environment variables
TOKEN_URL = os.getenv("TOKEN_URL")
CLIENTID = os.getenv("CLIENTID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
MODEL_ID = os.getenv("MODEL_ID", "text-reranking")


@OpeaComponentRegistry.register("OPEA_OVMS_RERANKING")
class OpeaOVMSReranking(OpeaComponent):
    """A specialized reranking component derived from OpeaComponent for OVMS reranking services.

    Attributes:
        client (AsyncInferenceClient): An instance of the client for reranking generation.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RERANK.name.lower(), description, config)
        self.base_url = os.getenv("OVMS_RERANKING_ENDPOINT", "http://localhost:8808")
        self.client = self._initialize_client()
        health_status = self.check_health()
        if not health_status:
            logger.error("OPEAOVMSReranking health check failed.")

    def _initialize_client(self) -> AsyncInferenceClient:
        """Initializes the AsyncInferenceClient."""
        access_token = (
            get_access_token(TOKEN_URL, CLIENTID, CLIENT_SECRET) if TOKEN_URL and CLIENTID and CLIENT_SECRET else None
        )
        headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
        return AsyncInferenceClient(
            model=MODEL_ID,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            headers=headers,
        )

    async def invoke(
        self, input: Union[SearchedDoc, RerankingRequest, ChatCompletionRequest]
    ) -> Union[LLMParamsDoc, RerankingResponse, ChatCompletionRequest]:
        """Invokes the reranking service to generate rerankings for the provided input."""
        reranking_results = []
        docs = []
        if input.retrieved_docs:
            docs = [doc.text for doc in input.retrieved_docs]
            if isinstance(input, SearchedDoc):
                query = input.initial_query
            else:
                # for RerankingRequest, ChatCompletionRequest
                query = input.input
            response = await self.client.post(
                json={"model": self.client.model, "query": query, "documents": docs, "top_n": input.top_n},
                model=f"{self.base_url}/v3/rerank",
                task="text-reranking",
            )
            for best_response in json.loads(response.decode())["results"]:
                reranking_results.append(
                    {
                        "text": input.retrieved_docs[best_response["index"]].text,
                        "score": best_response["relevance_score"],
                    }
                )

        if isinstance(input, SearchedDoc):
            result = [doc["text"] for doc in reranking_results]
            if logflag:
                logger.info(result)
            return LLMParamsDoc(query=input.initial_query, documents=result)
        else:
            reranking_docs = []
            for doc in reranking_results:
                reranking_docs.append(RerankingResponseData(text=docs[doc["index"]], score=doc["relevance_score"]))
            if isinstance(input, RerankingRequest):
                result = RerankingResponse(reranked_docs=reranking_docs)
                if logflag:
                    logger.info(result)
                return result

            if isinstance(input, ChatCompletionRequest):
                input.reranked_docs = reranking_docs
                input.documents = [doc["text"] for doc in reranking_results]
                if logflag:
                    logger.info(input)
                return input

    def check_health(self) -> bool:
        """Checks the health of the embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/v2/health/ready")
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            # Handle connection errors, timeouts, etc.
            logger.error(f"Health check failed: {e}")
        return False
