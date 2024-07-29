# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import Redis
from langsmith import traceable
from redis_config import EMBED_MODEL, INDEX_NAME, REDIS_URL

from comps import ServiceType, opea_microservices, register_microservice, register_statistics, statistics_dict
from comps.cores.proto.api_protocol import RetrievalRequest, RetrievalResponse, RetrievalResponseData

tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT")


@register_microservice(
    name="opea_service@retriever_redis",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
    input_datatype=RetrievalRequest,
    output_datatype=RetrievalResponse,
)
@traceable(run_type="retriever")
@register_statistics(names=["opea_service@retriever_redis"])
def retrieve(request: RetrievalRequest) -> RetrievalResponse:
    start = time.time()

    # check if the Redis index has data
    if vector_db.client.keys() == []:
        response = RetrievalResponse(retrieved_docs=[])
        statistics_dict["opea_service@retriever_redis"].append_latency(time.time() - start, None)
        return response

    if isinstance(request.embedding, list):
        embed = request.embedding
    else:
        # parse from EmbeddingResponse
        embed = request.embedding.data[0].embedding

    # if the Redis index has data, perform the search
    if request.search_type == "similarity":
        search_res = vector_db.similarity_search_by_vector(embedding=embed, k=request.k)
    elif request.search_type == "similarity_distance_threshold":
        if request.distance_threshold is None:
            raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
        search_res = vector_db.similarity_search_by_vector(
            embedding=embed, k=request.k, distance_threshold=request.distance_threshold
        )
    elif request.search_type == "similarity_score_threshold":
        docs_and_similarities = vector_db.similarity_search_with_relevance_scores(
            query=request.text, k=request.k, score_threshold=request.score_threshold
        )
        search_res = [doc for doc, _ in docs_and_similarities]
    elif request.search_type == "mmr":
        search_res = vector_db.max_marginal_relevance_search(
            query=request.text, k=request.k, fetch_k=request.fetch_k, lambda_mult=request.lambda_mult
        )
    searched_docs = []
    for r in search_res:
        searched_docs.append(RetrievalResponseData(text=r.page_content, metadata=r.metadata))
    response = RetrievalResponse(retrieved_docs=searched_docs)
    statistics_dict["opea_service@retriever_redis"].append_latency(time.time() - start, None)
    return response


if __name__ == "__main__":
    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    vector_db = Redis(embedding=embeddings, index_name=INDEX_NAME, redis_url=REDIS_URL)
    opea_microservices["opea_service@retriever_redis"].start()
