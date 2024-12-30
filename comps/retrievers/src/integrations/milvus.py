# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings, OpenAIEmbeddings

from comps import CustomLogger, EmbedDoc, OpeaComponent, SearchedDoc, ServiceType
from comps.vectorstores.src.opea_vectorstores_controller import OpeaVectorstoresController
from comps.vectorstores.src.integrations.milvus import OpeaMilvusVectorstores

from .config import (
    LOCAL_EMBEDDING_MODEL,
    MOSEC_EMBEDDING_ENDPOINT,
    MOSEC_EMBEDDING_MODEL,
    TEI_EMBEDDING_ENDPOINT,
)

logger = CustomLogger("milvus_retrievers")
logflag = os.getenv("LOGFLAG", False)


class MosecEmbeddings(OpenAIEmbeddings):
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        batched_embeddings: List[List[float]] = []
        response = self.client.create(input=texts, **self._invocation_params)
        if not isinstance(response, dict):
            response = response.model_dump()
        batched_embeddings.extend(r["embedding"] for r in response["data"])

        _cached_empty_embedding: Optional[List[float]] = None

        def empty_embedding() -> List[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = self.client.create(input="", **self._invocation_params)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else empty_embedding() for e in batched_embeddings]


class OpeaMilvusRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for milvus retriever services.

    Attributes:
        client (Milvus): An instance of the milvus client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.db_controller = self._initialize_db_controller()

    def _initialize_embedder(self):
        if MOSEC_EMBEDDING_ENDPOINT:
            # create embeddings using Mosec endpoint service
            if logflag:
                logger.info(f"[ init embedder ] MOSEC_EMBEDDING_ENDPOINT:{MOSEC_EMBEDDING_ENDPOINT}")
            embeddings = MosecEmbeddings(model=MOSEC_EMBEDDING_MODEL)
        elif TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            embeddings = HuggingFaceHubEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{LOCAL_EMBEDDING_MODEL}")
            embeddings = HuggingFaceBgeEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        return embeddings

    def _initialize_db_controller(self) -> OpeaVectorstoresController:
        controller = OpeaVectorstoresController()
        milvus_db = OpeaMilvusVectorstores(
            embedder=self.embedder,
            name="OpeaMilvusVectorstore",
            description="OPEA Milvus Vectorstore Service"
        )
        controller.register(milvus_db)
        controller.discover_and_activate()
        return controller

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of milvus")
        try:
            if self.db_controller.active_component.check_health():
                if logflag:
                    logger.info("[ health check ] Successfully connected to milvus!")
                return True
        except Exception as e:
            logger.info(f"[ health check ] Failed to connect to milvus: {e}")
            return False

    async def invoke(self, input: EmbedDoc) -> SearchedDoc:
        """Search the Milvus index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            Union[SearchedDoc, RetrievalResponse, ChatCompletionRequest]: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        if self.db_controller.is_empty():
            search_res = []
        else:
            search_res = await self.db_controller.similarity_search(
                input=input.text,
                embedding=input.embedding,
                search_type=input.search_type,
                k=input.k,
                distance_threshold=input.distance_threshold,
                score_threshold=input.score_threshold,
                lambda_mult=input.lambda_mult
            )

        if logflag:
            logger.info(f"retrieve result: {search_res}")

        return search_res
