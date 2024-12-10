# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from sentence_transformers import CrossEncoder

from comps import CustomLogger, RerankedDoc, SearchedDoc, ServiceType, opea_microservices, register_microservice

logger = CustomLogger("local_reranking")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@local_reranking",
    service_type=ServiceType.RERANK,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=8000,
    input_datatype=SearchedDoc,
    output_datatype=RerankedDoc,
)
def reranking(input: SearchedDoc) -> RerankedDoc:
    if logflag:
        logger.info(input)
    query_and_docs = [(input.initial_query, doc.text) for doc in input.retrieved_docs]
    scores = reranker_model.predict(query_and_docs)
    first_passage = sorted(list(zip(input.retrieved_docs, scores)), key=lambda x: x[1], reverse=True)[0][0]
    res = RerankedDoc(initial_query=input.initial_query, reranked_docs=[first_passage])
    if logflag:
        logger.info(res)
    return res


if __name__ == "__main__":
    reranker_model = CrossEncoder(model_name="BAAI/bge-reranker-base", max_length=512)
    opea_microservices["opea_service@local_reranking"].start()
