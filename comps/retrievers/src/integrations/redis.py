# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Union

from fastapi import HTTPException
from langchain.vectorstores import Redis
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from comps import (
    CustomLogger,
    EmbedDoc,
    EmbedMultimodalDoc,
    OpeaComponent,
    OpeaComponentRegistry,
    SearchedDoc,
    ServiceType,
)
from comps.cores.proto.api_protocol import ChatCompletionRequest, EmbeddingResponse, RetrievalRequest, RetrievalResponse

from .config import (
    BRIDGE_TOWER_EMBEDDING,
    EMBED_MODEL,
    HUGGINGFACEHUB_API_TOKEN,
    INDEX_NAME,
    INDEX_SCHEMA,
    REDIS_URL,
    TEI_EMBEDDING_ENDPOINT,
)

logger = CustomLogger("redis_retrievers")
logflag = os.getenv("LOGFLAG", False)
executor = ThreadPoolExecutor()


async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))


@OpeaComponentRegistry.register("OPEA_RETRIEVER_REDIS")
class OpeaRedisRetriever(OpeaComponent):
    """A specialized retriever component derived from OpeaComponent for redis retriever services.

    Attributes:
        client (redis.Redis): An instance of the redis client for vector database operations.
    """

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.RETRIEVER.name.lower(), description, config)
        self.embeddings = asyncio.run(self._initialize_embedder())
        self.client = asyncio.run(self._initialize_client())
        health_status = self.check_health()
        if not health_status:
            logger.error("OpeaRedisRetriever health check failed.")

    async def _initialize_embedder(self):
        if TEI_EMBEDDING_ENDPOINT:
            logger.info("use tei embedding")
            if not HUGGINGFACEHUB_API_TOKEN:
                raise HTTPException(
                    status_code=400,
                    detail="You MUST offer the `HUGGINGFACEHUB_API_TOKEN` when using `TEI_EMBEDDING_ENDPOINT`.",
                )

            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(TEI_EMBEDDING_ENDPOINT + "/info")
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400, detail=f"TEI embedding endpoint {TEI_EMBEDDING_ENDPOINT} is not available."
                    )
                model_id = response.json()["model_id"]
            # create embeddings using TEI endpoint service
            embedder = HuggingFaceInferenceAPIEmbeddings(
                api_key=HUGGINGFACEHUB_API_TOKEN, model_name=model_id, api_url=TEI_EMBEDDING_ENDPOINT
            )
        elif BRIDGE_TOWER_EMBEDDING:
            logger.info("use bridge tower embedding")
            from comps.third_parties.bridgetower.src.bridgetower_embedding import BridgeTowerEmbedding

            embedder = BridgeTowerEmbedding()
        else:
            logger.info("use local embedding")
            # create embeddings using local embedding model
            embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return embedder

    async def _initialize_client(self, index_name=INDEX_NAME) -> Redis:
        """Initializes the redis client."""
        try:
            if BRIDGE_TOWER_EMBEDDING:
                logger.info(f"generate multimodal redis instance with {BRIDGE_TOWER_EMBEDDING}")
                client = Redis(
                    embedding=self.embeddings, index_name=index_name, index_schema=INDEX_SCHEMA, redis_url=REDIS_URL
                )
            else:
                client = Redis(embedding=self.embeddings, index_name=index_name, redis_url=REDIS_URL)
            return client
        except Exception as e:
            logger.error(f"fail to initialize redis client: {e}")
            return None

    def check_health(self) -> bool:
        """Checks the health of the retriever service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        if logflag:
            logger.info("[ health check ] start to check health of redis")
        try:
            if self.client:
                self.client.client.ping()
                if logflag:
                    logger.info("[ health check ] Successfully connected to Redis!")
                return True
        except Exception as e:
            logger.info(f"[ health check ] Failed to connect to Redis: {e}")
            return False

    async def invoke(
        self, input: Union[EmbedDoc, EmbedMultimodalDoc, RetrievalRequest, ChatCompletionRequest]
    ) -> Union[SearchedDoc, RetrievalResponse, ChatCompletionRequest]:
        """Search the Redis index for the most similar documents to the input query.

        Args:
            input (Union[EmbedDoc, RetrievalRequest, ChatCompletionRequest]): The input query to search for.
        Output:
            Union[SearchedDoc, RetrievalResponse, ChatCompletionRequest]: The retrieved documents.
        """
        if logflag:
            logger.info(input)

        client = self.client
        if isinstance(input, EmbedDoc) and input.index_name and input.index_name != INDEX_NAME:
            client = asyncio.run(self._initialize_client(index_name=input.index_name))

        # check if the Redis index has data
        try:
            keys_exist = client.client.keys()

        except Exception as e:
            logger.error(f"Redis key check failed: {e}")
            keys_exist = []

        if not keys_exist:
            if logflag:
                logger.info("No data in Redis index, return []")
            search_res = []
        else:
            if isinstance(input, EmbedDoc) or isinstance(input, EmbedMultimodalDoc):
                embedding_data_input = input.embedding
            else:
                # for RetrievalRequest, ChatCompletionRequest
                if isinstance(input.embedding, EmbeddingResponse):
                    embedding_data_input = [emb.embedding for emb in input.embedding.data]

                else:
                    embedding_data_input = input.embedding

            # if the Redis index has data, perform the search
            if input.search_type == "similarity":
                search_res = await run_in_thread(
                    client.similarity_search_by_vector, embedding=embedding_data_input, k=input.k
                )
            elif input.search_type == "similarity_distance_threshold":
                if input.distance_threshold is None:
                    raise ValueError(
                        "distance_threshold must be provided for " + "similarity_distance_threshold retriever"
                    )
                search_res = await run_in_thread(
                    client.similarity_search_by_vector,
                    embedding=input.embedding,
                    k=input.k,
                    distance_threshold=input.distance_threshold,
                )
            elif input.search_type == "similarity_score_threshold":
                docs_and_similarities = await run_in_thread(
                    client.similarity_search_with_relevance_scores,
                    query=input.text,
                    k=input.k,
                    score_threshold=input.score_threshold,
                )
                search_res = [doc for doc, _ in docs_and_similarities]
            elif input.search_type == "mmr":
                search_res = await run_in_thread(
                    client.max_marginal_relevance_search,
                    query=input.text,
                    k=input.k,
                    fetch_k=input.fetch_k,
                    lambda_mult=input.lambda_mult,
                )
            else:
                raise ValueError(f"{input.search_type} not valid")

        if logflag:
            logger.info(search_res)

        return search_res
