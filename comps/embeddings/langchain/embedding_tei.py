# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langsmith import traceable

from comps import (
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import EmbeddingRequest, EmbeddingResponse, EmbeddingResponseData


@register_microservice(
    name="opea_service@embedding_tei_langchain",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    input_datatype=EmbeddingRequest,
    output_datatype=EmbedDoc768,
)
@traceable(run_type="embedding")
@register_statistics(names=["opea_service@embedding_tei_langchain"])
def embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    start = time.time()
    embed_vector = embeddings.embed_query(request.input)
    embed_vector = embed_vector[:request.dimensions]
    response = EmbeddingResponse(data=[EmbeddingResponseData(index=0, embedding=embed_vector)])
    statistics_dict["opea_service@embedding_tei_langchain"].append_latency(time.time() - start, None)
    return response


if __name__ == "__main__":
    tei_embedding_endpoint = os.getenv("TEI_EMBEDDING_ENDPOINT", "http://localhost:8080")
    embeddings = HuggingFaceHubEmbeddings(model=tei_embedding_endpoint)
    print("TEI Gaudi Embedding initialized.")
    opea_microservices["opea_service@embedding_tei_langchain"].start()
