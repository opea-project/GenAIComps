# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from comps import BridgeTowerEmbedding
from langchain_community.vectorstores import Redis
from langsmith import traceable
from multimodal_config import INDEX_NAME, REDIS_URL, REDIS_SCHEMA

from comps import (
    EmbedMultimodalDoc,
    SearchedMultimodalDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResponseData,
)

@register_microservice(
    name="opea_service@retriever_redis",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@traceable(run_type="retriever")
@register_statistics(names=["opea_service@retriever_redis"])
def retrieve(
    input: Union[EmbedMultimodalDoc, RetrievalRequest, ChatCompletionRequest]
) -> Union[SearchedMultimodalDoc, RetrievalResponse, ChatCompletionRequest]:

    start = time.time()
    # check if the Redis index has data
    if vector_db.client.keys() == []:
        search_res = []
    else:
        if isinstance(input, EmbedMultimodalDoc):
            query = input.text
        else:
            # for RetrievalRequest, ChatCompletionRequest
            query = input.input
        # if the Redis index has data, perform the search
        if input.search_type == "similarity":
            search_res = vector_db.similarity_search_by_vector(embedding=input.embedding, k=input.k)
        elif input.search_type == "similarity_distance_threshold":
            if input.distance_threshold is None:
                raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
            search_res = vector_db.similarity_search_by_vector(
                embedding=input.embedding, k=input.k, distance_threshold=input.distance_threshold
            )
        elif input.search_type == "similarity_score_threshold":
            docs_and_similarities = vector_db.similarity_search_with_relevance_scores(
                query=input.text, k=input.k, score_threshold=input.score_threshold
            )
            search_res = [doc for doc, _ in docs_and_similarities]
        elif input.search_type == "mmr":
            search_res = vector_db.max_marginal_relevance_search(
                query=input.text, k=input.k, fetch_k=input.fetch_k, lambda_mult=input.lambda_mult
            )
        else:
            raise ValueError(f"{input.search_type} not valid")

    # return different response format
    retrieved_docs = []
    if isinstance(input, EmbedMultimodalDoc):
        metadata_list = []
        for r in search_res:
            metadata_list.append(r.metadata)
            retrieved_docs.append(TextDoc(text=r.page_content))
        result = SearchedMultimodalDoc(retrieved_docs=retrieved_docs, initial_query=input.text, metadata=metadata_list)
    else:
        for r in search_res:
            retrieved_docs.append(RetrievalResponseData(text=r.page_content, metadata=r.metadata))
        if isinstance(input, RetrievalRequest):
            result = RetrievalResponse(retrieved_docs=retrieved_docs)
        elif isinstance(input, ChatCompletionRequest):
            input.retrieved_docs = retrieved_docs
            input.documents = [doc.text for doc in retrieved_docs]
            result = input

    statistics_dict["opea_service@retriever_redis"].append_latency(time.time() - start, None)
    return result


if __name__ == "__main__":

    embeddings = BridgeTowerEmbedding()
    vector_db = Redis.from_existing_index(embedding=embeddings, schema=REDIS_SCHEMA, index_name=INDEX_NAME, redis_url=REDIS_URL)
    opea_microservices["opea_service@retriever_redis"].start()