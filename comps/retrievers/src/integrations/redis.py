# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Union

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from comps import (
    CustomLogger, EmbedDoc, OpeaComponent, SearchedDoc, ServiceType,
    EmbedMultimodalDoc, SearchedMultimodalDoc,
)
from comps.embeddings.src.integrations.dependency.bridgetower import BridgeTowerEmbedding
from comps.cores.proto.api_protocol import ChatCompletionRequest, EmbeddingResponse, RetrievalRequest, RetrievalResponse
from comps.vectorstores.src.opea_vectorstores_controller import OpeaVectorstoresController
from comps.vectorstores.src.integrations.redis import OpeaRedisVectorstores

from .config import EMBED_MODEL,TEI_EMBEDDING_ENDPOINT, BRIDGE_TOWER_EMBEDDING

logger = CustomLogger("redis_retrievers")
logflag = os.getenv("LOGFLAG", False)


class OpeaRedisRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for redis retriever services.

    Attributes:
        client (redis.Redis): An instance of the redis client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        # Create embeddings
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            self.embedder = HuggingFaceEndpointEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        elif BRIDGE_TOWER_EMBEDDING:
            logger.info(f"use bridge tower embedding")
            self.embedder = BridgeTowerEmbedding()
        else:
            # create embeddings using local embedding model
            self.embedder = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
        self.db_controller = self._initialize_db_controller()
        
    def _initialize_db_controller(self) -> OpeaVectorstoresController:
        controller = OpeaVectorstoresController()
        redis_db = OpeaRedisVectorstores(
            embedder=self.embedder,
            name="OpeaRedisVectorstore",
            description="OPEA Redis Vectorstore Service",
            is_multimodal=BRIDGE_TOWER_EMBEDDING
        )
        controller.register(redis_db)
        controller.discover_and_activate()
        return controller

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of redis")
        try:
            if self.db_controller.active_component.check_health():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Redis!")
                return True
        except Exception as e:
            logger.info(f"[ health check ] Failed to connect to Redis: {e}")
            return False

    async def invoke(
            self, 
            input: Union[EmbedDoc, EmbedMultimodalDoc, RetrievalRequest, ChatCompletionRequest]
    ):
        """Search the Redis index for the most similar documents to the input query.

        Args:
            input (Union[EmbedDoc, RetrievalRequest, ChatCompletionRequest]): The input query to search for.
        Output:
            Union[SearchedDoc, RetrievalResponse, ChatCompletionRequest]: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        # check if the Redis index has data
        if self.db_controller.is_empty():
            search_res = []
        else:
            if isinstance(input, EmbedDoc) or isinstance(input, EmbedMultimodalDoc):
                embedding_data_input = input.embedding
            else:
                # for RetrievalRequest, ChatCompletionRequest
                if isinstance(input.embedding, EmbeddingResponse):
                    embeddings = input.embedding.data
                    embedding_data_input = []
                    for emb in embeddings:
                        embedding_data_input.append(emb.embedding)
                else:
                    embedding_data_input = input.embedding

            search_res = await self.db_controller.similarity_search(
                input=input.text,
                embedding=embedding_data_input,
                search_type=input.search_type,
                k=input.k,
                distance_threshold=input.distance_threshold,
                score_threshold=input.score_threshold,
                lambda_mult=input.lambda_mult
            )

        if logflag:
            logger.info(search_res)

        return search_res
