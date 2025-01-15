# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.vector_stores.redis import RedisVectorStore
from redis_config import REDIS_URL

from comps import CustomLogger, EmbedDoc, SearchedDoc, ServiceType, TextDoc, opea_microservices, register_microservice

logger = CustomLogger("retriever_redis")
logflag = os.getenv("LOGFLAG", False)

tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT")


@register_microservice(
    name="opea_service@retriever_redis",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
async def retrieve(input: EmbedDoc) -> SearchedDoc:
    if logflag:
        logger.info(input)
    vector_store_query = VectorStoreQuery(query_embedding=input.embedding)
    search_res = await vector_store.aquery(query=vector_store_query)
    searched_docs = []
    for node, id, similarity in zip(search_res.nodes, search_res.ids, search_res.similarities):
        searched_docs.append(TextDoc(text=node.get_content()))
    result = SearchedDoc(retrieved_docs=searched_docs, initial_query=input.text)
    if logflag:
        logger.info(result)
    return result


if __name__ == "__main__":

    vector_store = RedisVectorStore(
        redis_url=REDIS_URL,
    )
    opea_microservices["opea_service@retriever_redis"].start()
