# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceHubEmbeddings
from langchain_community.vectorstores import PathwayVectorClient
# from langsmith import traceable


from comps import (
    EmbedDoc768,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)



# from pathway.xpacks.llm.vector_store import VectorStoreServer
# from pathway.xpacks.llm.parsers import ParseUnstructured
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter

# data = pw.io.fs.read(
#     "./data",
#     format="binary",
#     mode="streaming",
#     with_metadata=True,
# )
# from unittest.mock import MagicMock
# embeddings = OpenAIEmbeddings(api_key="api_key")
# embeddings.aembed_documents = MagicMock(return_value=[1.0, 2.0, 3.0, 0.1])
# splitter = CharacterTextSplitter()

host = os.getenv("PATHWAY_HOST", "127.0.0.1")
port = int(os.getenv("PATHWAY_PORT", 8666))
print(host, port, "HOST AND PORTS")


EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT") == 10


# @traceable(run_type="retriever")
@register_microservice(
    name="opea_service@retriever_pathway",
    service_type=ServiceType.RETRIEVER,
    endpoint="/v1/retrieval",
    host="0.0.0.0",
    port=7000,
)
@register_statistics(names=["opea_service@retriever_pathway"])
def retrieve(input: EmbedDoc768) -> SearchedDoc:
    start = time.time()
    documents = pw_client.similarity_search(input.text, input.fetch_k)

    docs = [TextDoc(text=r.page_content) for r in documents]

    statistics_dict["opea_service@retriever_redis"].append_latency(time.time() - start, None)
    return SearchedDoc(retrieved_docs=docs, initial_query=input.text)


if __name__ == "__main__":
    # Create vectorstore
    if tei_embedding_endpoint:
        # create embeddings using TEI endpoint service
        embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    else:
        # create embeddings using local embedding model
        embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    # vector_db = Redis(embedding=embeddings, index_name=INDEX_NAME, redis_url=REDIS_URL)
    # server = VectorStoreServer.from_langchain_components(
    #     data, embedder=embeddings, parser=ParseUnstructured(), splitter=splitter
    # )
    # server.run_server(host, port=port, with_cache=True, cache_backend=pw.persistence.Backend.filesystem("./Cache"))  # , threaded=True
    pw_client = PathwayVectorClient(host=host, port=port)
    opea_microservices["opea_service@retriever_pathway"].start()
