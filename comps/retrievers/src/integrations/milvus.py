# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os

from fastapi import HTTPException
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus.vectorstores import Milvus

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType

from .config import (
    BRIDGE_TOWER_EMBEDDING,
    COLLECTION_NAME,
    HF_TOKEN,
    INDEX_PARAMS,
    LOCAL_EMBEDDING_MODEL,
    MILVUS_URI,
    TEI_EMBEDDING_ENDPOINT,
)

logger = CustomLogger("milvus_retrievers")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_RETRIEVER_MILVUS")
class OpeaMilvusRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for milvus retriever services.

    Attributes:
        client (Milvus): An instance of the milvus client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaMilvusRetriever health check failed.")

    def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            # create embeddings using TEI endpoint service
            if logflag:
                logger.info(f"[ init embedder ] TEI_EMBEDDING_ENDPOINT:{TEI_EMBEDDING_ENDPOINT}")
            if not HF_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HF_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )
            import requests

            response = requests.get(TEI_EMBEDDING_ENDPOINT + "/info")
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                )
            model_id = response.json()["model_id"]
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=HF_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        elif BRIDGE_TOWER_EMBEDDING:
            logger.info("use bridge tower embedding")
            from comps.third_parties.bridgetower.src.bridgetower_embedding import BridgeTowerEmbedding

            embeddings = BridgeTowerEmbedding()
        else:
            # create embeddings using local embedding model
            if logflag:
                logger.info(f"[ init embedder ] LOCAL_EMBEDDING_MODEL:{LOCAL_EMBEDDING_MODEL}")
            embeddings = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        return embeddings

    def _initialize_client(self) -> Milvus:
        """Initializes the milvus client."""
        try:
            client = Milvus(
                embedding_function=self.embedder,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": MILVUS_URI},
                index_params=INDEX_PARAMS,
                auto_id=True,
            )
            return client
        except Exception as e:
            logger.error(f"fail to initialize milvus client: {e}")
            return None

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of milvus")
        try:
            client = Milvus(
                embedding_function=self.embedder,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": MILVUS_URI},
                index_params=INDEX_PARAMS,
                auto_id=True,
            )
            _ = client.client.list_collections()
            if logflag:
                logger.info("[ check health ] Successfully connected to Milvus!")
            return True
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to Milvus: {e}")
            return False

    async def invoke(self, input: EmbedDoc) -> list:
        """Search the Milvus index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            list: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        my_milvus = Milvus(
            embedding_function=self.embedder,
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            index_params=INDEX_PARAMS,
            auto_id=True,
        )

        if input.search_type == "similarity":
            search_res = await my_milvus.asimilarity_search_by_vector(embedding=input.embedding, k=input.k)
        elif input.search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
            search_res = await my_milvus.asimilarity_search_by_vector(
                embedding=input.embedding, k=input.k, distance_threshold=input.distance_threshold
            )
        elif input.search_type == "similarity_score_threshold":
            docs_and_similarities = await my_milvus.asimilarity_search_with_relevance_scores(
                query=input.text, k=input.k, score_threshold=input.score_threshold
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif input.search_type == "mmr":
            search_res = await my_milvus.amax_marginal_relevance_search(
                query=input.text, k=input.k, fetch_k=input.fetch_k, lambda_mult=input.lambda_mult
            )

        if logflag:
            logger.info(f"retrieve result: {search_res}")

        return search_res
