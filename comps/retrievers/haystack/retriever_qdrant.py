# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from langsmith import traceable
from haystack import Pipeline
from haystack.components.embedders import HuggingFaceTEITextEmbedder, SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from qdrant_config import EMBED_MODEL, INDEX_NAME, EMBED_DIMENSION, QDRANT_URL

from comps import SearchedDoc, ServiceType, TextDoc, opea_microservices, register_microservice

tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT")

# Create a pipeline for querying a Qdrant document store 
def initialize_query_pipeline() -> Pipeline:
    qdrant_store = QdrantDocumentStore(
        url=QDRANT_URL,
        embedding_dim=EMBED_DIMENSION,
        index=INDEX_NAME,
        recreate_index=False
    )
    
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embedder = HuggingFaceTEITextEmbedder(url=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embedder = SentenceTransformersTextEmbedder(model=EMBED_MODEL)
        embedder.warm_up()

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", embedder)
    query_pipeline.add_component("retriever", QdrantEmbeddingRetriever(qdrant_store))
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    return query_pipeline

@register_microservice(
    name="opea_service@retriever_qdrant",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@traceable(run_type="retriever")
def retrieve(input: TextDoc) -> SearchedDoc:
    search_res = query_pipeline.run({"text_embedder": {"text": input.text}})['retriever']['documents']
    searched_docs = [TextDoc(text=r.content) for r in search_res]
    result = SearchedDoc(retrieved_docs=searched_docs, initial_query=input.text)
    return result


if __name__ == "__main__":
    query_pipeline = initialize_query_pipeline()
    opea_microservices["opea_service@retriever_qdrant"].start()
