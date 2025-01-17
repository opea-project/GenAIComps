# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
import time

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores.vdms import VDMS, VDMS_Client

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import (
    DISTANCE_STRATEGY,
    EMBED_MODEL,
    SEARCH_ENGINE,
    TEI_EMBEDDING_ENDPOINT,
    VDMS_HOST,
    VDMS_INDEX_NAME,
    VDMS_PORT,
    VDMS_USE_CLIP,
)

logger = CustomLogger("vdms_retrievers")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_RETRIEVER_VDMS")
class OpeaVDMsRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for vdms retriever services.

    Attributes:
        client (VDMs): An instance of the vdms client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.client = VDMS_Client(VDMS_HOST, VDMS_PORT)
        self.vector_db = self._initialize_vector_db()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaVDMsRetriever health check failed.")

    def _initialize_embedder(self):
        if VDMS_USE_CLIP:
            from comps.third_parties.clip.src.clip_embedding import vCLIP

            embeddings = vCLIP({"model_name": "openai/clip-vit-base-patch32", "num_frm": 64})
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            embeddings = HuggingFaceHubEmbeddings(model=TEI_EMBEDDING_ENDPOINT)
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{EMBED_MODEL}")
            embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
        return embeddings

    def _initialize_vector_db(self) -> VDMS:
        """Initializes the vdms client."""
        if VDMS_USE_CLIP:
            dimensions = self.embedder.get_embedding_length()
            vector_db = VDMS(
                client=self.client,
                embedding=self.embedder,
                collection_name=VDMS_INDEX_NAME,
                embedding_dimensions=dimensions,
                distance_strategy=DISTANCE_STRATEGY,
                engine=SEARCH_ENGINE,
            )
        else:
            vector_db = VDMS(
                client=self.client,
                embedding=self.embedder,
                collection_name=VDMS_INDEX_NAME,
                distance_strategy=DISTANCE_STRATEGY,
                engine=SEARCH_ENGINE,
            )
        return vector_db

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of vdms")
        try:
            if self.vector_db:
                logger.info("[ check health ] Successfully connected to VDMs!")
                return True
            else:
                logger.info("[ check health ] Failed to connect to VDMs.")
                return False
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to VDMs: {e}")
            return False

    async def invoke(self, input: EmbedDoc) -> list:
        """Search the VDMs index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            list: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        if input.search_type == "similarity":
            search_res = self.vector_db.similarity_search_by_vector(
                embedding=input.embedding, k=input.k, filter=input.constraints
            )
        elif input.search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
            search_res = self.vector_db.similarity_search_by_vector(
                embedding=input.embedding,
                k=input.k,
                distance_threshold=input.distance_threshold,
                filter=input.constraints,
            )
        elif input.search_type == "similarity_score_threshold":
            docs_and_similarities = self.vector_db.similarity_search_with_relevance_scores(
                query=input.text, k=input.k, score_threshold=input.score_threshold, filter=input.constraints
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif input.search_type == "mmr":
            search_res = self.vector_db.max_marginal_relevance_search(
                query=input.text,
                k=input.k,
                fetch_k=input.fetch_k,
                lambda_mult=input.lambda_mult,
                filter=input.constraints,
            )

        if logflag:
            logger.info(f"retrieve result: {search_res}")

        return search_res
