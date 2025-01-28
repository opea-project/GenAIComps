# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from predictionguard import PredictionGuard

from comps import (
    GeneratedDoc,
    LLMParamsDoc,
    RerankedDoc,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.reranks.predictionguard.src.helpers import process_doc_list

client = PredictionGuard()
app = FastAPI()


@register_microservice(
    name="opea_service@reranks_predictionguard",
    service_type=ServiceType.LLM,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=9000,
    input_datatype=SearchedDoc,
    output_datatype=RerankedDoc,
)
@register_statistics(names=["opea_service@reranks_predictionguard"])
def reranks_generate(input: SearchedDoc) -> RerankedDoc:
    start = time.time()
    reranked_docs = []

    if input.retrieved_docs:
        docs = process_doc_list(input.retrieved_docs)

        try:
            rerank_result = client.rerank.create(
                model="bge-reranker-v2-m3", query=input.initial_query, documents=docs, return_documents=True
            )

            # based on rerank_result, reorder the retrieved_docs to match the order of the retrieved_docs in the input
            reranked_docs = [
                TextDoc(id=input.retrieved_docs[doc["index"]].id, text=doc["text"]) for doc in rerank_result["results"]
            ]

        except ValueError as e:
            logging.error(f"rerank failed with error: {e}. Inputs: query={input.initial_query}, documents={docs}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    else:
        logging.info("reranking request input did not contain any documents")

    statistics_dict["opea_service@reranks_predictionguard"].append_latency(time.time() - start, None)
    return RerankedDoc(initial_query=input.initial_query, reranked_docs=reranked_docs)


if __name__ == "__main__":
    opea_microservices["opea_service@reranks_predictionguard"].start()
