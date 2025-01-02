# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from comps import CustomLogger, EmbedDoc, OpeaComponent, SearchedDoc, ServiceType
from comps.vectorstores.src.integrations.pinecone import OpeaPineconeVectorstores
from comps.vectorstores.src.opea_vectorstores_controller import OpeaVectorstoresController

from .config import EMBED_MODEL, TEI_EMBEDDING_ENDPOINT

logger = CustomLogger("pinecone_retrievers")
logflag = os.getenv("LOGFLAG", False)


class OpeaPineconeRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for milvus retriever services.

    Attributes:
        client (Pinecone): An instance of the milvus client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.db_controller = self._initialize_db_controller()

    def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            embeddings = HuggingFaceEndpointEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_db_controller(self) -> OpeaVectorstoresController:
        controller = OpeaVectorstoresController()
        milvus_db = OpeaPineconeVectorstores(
            embedder=self.embedder, name="OpeaPineconeVectorstore", description="OPEA Pinecone Vectorstore Service"
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
            logger.info("[ health check ] start to check health of Pinecone")
        try:
            if self.db_controller.active_component.check_health():
                if logflag:
                    logger.info("[ health check ] Successfully connected to Pinecone!")
                return True
        except Exception as e:
            logger.info(f"[ health check ] Failed to connect to Pinecone: {e}")
            return False

    async def invoke(self, input: EmbedDoc) -> SearchedDoc:
        """Search the Pinecone index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
            Output (SearchedDoc): The search results.
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
                fetch_k=input.fetch_k,
                distance_threshold=input.distance_threshold,
                score_threshold=input.score_threshold,
                lambda_mult=input.lambda_mult,
            )

        if logflag:
            logger.info(f"retrieve result: {search_res}")

        return search_res
