# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores.vdms import VDMS, VDMS_Client
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langsmith import traceable
from vdms_config import (  # , HUGGINGFACEHUB_API_TOKEN, INDEX_SCHEMA, VDMS_URL
    COLLECTION_NAME,
    DISTANCE_STRATEGY,
    EMBED_MODEL,
    SEARCH_ENGINE,
    VDMS_HOST,
    VDMS_PORT,
)

from comps import (
    EmbedDoc,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Debugging
all_variables = dir()

for name in all_variables:

    # Print the item if it doesn't start with '__'
    if not name.startswith("__"):
        myvalue = eval(name)
        print(name, "is", type(myvalue), "and = ", myvalue)

# client = VDMS_Client(VDMS_HOST, VDMS_PORT)


# VDMS_HOST="172.17.0.2"
# VDMS_HOST="10.54.80.228"
# print("Host =", VDMS_HOST)
# end debugging

client = VDMS_Client(VDMS_HOST, VDMS_PORT)


@register_microservice(
    name="opea_service@retriever_vdms",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@traceable(run_type="retriever")
@register_statistics(names=["opea_service@retriever_vdms"])
def retrieve(input: EmbedDoc) -> SearchedDoc:
    start = time.time()
    constraints = None
    # place holder for adding constraints this has to be passed in the EmbedDoc input
    # so retriever can filter on them, if this functionality is needed

    if input.search_type == "similarity":
        search_res = vector_db.similarity_search_by_vector(embedding=input.embedding, k=input.k, filter=constraints)
    elif input.search_type == "similarity_distance_threshold":
        if input.distance_threshold is None:
            raise ValueError("distance_threshold must be provided for " + "similarity_distance_threshold retriever")
        search_res = vector_db.similarity_search_by_vector(
            embedding=input.embedding, k=input.k, distance_threshold=input.distance_threshold, filter=constraints
        )
    elif input.search_type == "similarity_score_threshold":
        docs_and_similarities = vector_db.similarity_search_with_relevance_scores(
            query=input.text, k=input.k, score_threshold=input.score_threshold, filter=constraints
        )
        search_res = [doc for doc, _ in docs_and_similarities]
    elif input.search_type == "mmr":
        search_res = vector_db.max_marginal_relevance_search(
            query=input.text, k=input.k, fetch_k=input.fetch_k, lambda_mult=input.lambda_mult, filter=constraints
        )
    searched_docs = []
    for r in search_res:
        searched_docs.append(TextDoc(text=r.page_content))
    result = SearchedDoc(retrieved_docs=searched_docs, initial_query=input.text)
    statistics_dict["opea_service@retriever_vdms"].append_latency(time.time() - start, None)
    return result


if __name__ == "__main__":
    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        # print(f"TEI_EMBEDDING_ENDPOINT:{tei_embedding_endpoint}")
        # embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint,huggingfacehub_api_token=hf_token)
        # embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
        embeddings = HuggingFaceEndpointEmbeddings(model=tei_embedding_endpoint, huggingfacehub_api_token=hf_token)
        # embeddings = HuggingFaceEndpointEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    # debug
    # embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
    # end debug

    vector_db = VDMS(
        client=client,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        embedding_dimensions=768,
        distance_strategy=DISTANCE_STRATEGY,
        engine=SEARCH_ENGINE,
    )
    opea_microservices["opea_service@retriever_vdms"].start()
