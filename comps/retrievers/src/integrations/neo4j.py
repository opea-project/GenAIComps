# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import os
from typing import Union

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import Neo4jVector

from comps import CustomLogger, EmbedDoc, OpeaComponent, OpeaComponentRegistry, ServiceType
from comps.cores.proto.api_protocol import ChatCompletionRequest, RetrievalRequest

from .config import EMBED_MODEL, NEO4J_PASSWORD, NEO4J_URL, NEO4J_USERNAME, TEI_EMBEDDING_ENDPOINT

logger = CustomLogger("neo4j_retrievers")
logflag = os.getenv("LOGFLAG", False)


@OpeaComponentRegistry.register("OPEA_RETRIEVER_NEO4J")
class OpeaNeo4jRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for neo4j retriever services.

    Attributes:
        client (Neo4j): An instance of the neo4j client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)

        self.embedder = self._initialize_embedder()
        self.neo4j_url = NEO4J_URL
        self.neo4j_username = NEO4J_USERNAME
        self.neo4j_password = NEO4J_PASSWORD
        self.vector_db = self._initialize_client()
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaNeo4jRetriever health check failed.")

    def _initialize_embedder(self):
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

    def _initialize_client(self):
        """Initializes the neo4j client."""
        vector_db = Neo4jVector.from_existing_graph(
            embedding=self.embedder,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            node_label="__Entity__",
            text_node_properties=["id", "description"],
            embedding_node_property="embedding",
        )
        return vector_db

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ check health ] start to check health of neo4j")
        try:
            result = self.vector_db.query("RETURN 1 AS result")
            logger.info(f"[ check health ] result: {result}")
            if result and result[0].get("result") == 1:
                logger.info("[ check health ] Successfully connected to Neo4j!")
                return True
            else:
                logger.info(f"[ check health ] Failed to connect to Neo4j")
                return False
        except Exception as e:
            logger.info(f"[ check health ] Failed to connect to Neo4j: {e}")
            return False

    async def invoke(self, input: Union[EmbedDoc, RetrievalRequest, ChatCompletionRequest]) -> list:
        """Search the Neo4j index for the most similar documents to the input query.

        Args:
            input (EmbedDoc): The input query to search for.
        Output:
            list: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        if isinstance(input, EmbedDoc):
            query = input.text
        else:
            # for RetrievalRequest, ChatCompletionRequest
            query = input.input

        if input.search_type == "similarity":
            search_res = await self.vector_db.asimilarity_search_by_vector(
                embedding=input.embedding, query=query, k=input.k
            )
        elif input.search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
            search_res = await self.vector_db.asimilarity_search_by_vector(
                embedding=input.embedding, query=query, k=input.k, distance_threshold=input.distance_threshold
            )
        elif input.search_type == "similarity_score_threshold":
            docs_and_similarities = await self.vector_db.asimilarity_search_with_relevance_scores(
                query=query, k=input.k, score_threshold=input.score_threshold
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif input.search_type == "mmr":
            search_res = await self.vector_db.amax_marginal_relevance_search(
                query=query, k=input.k, fetch_k=input.fetch_k, lambda_mult=input.lambda_mult
            )
        else:
            raise ValueError(f"{input.search_type} not valid")

        if logflag:
            logger.info(f"retrieve result: {search_res}")

        return search_res
