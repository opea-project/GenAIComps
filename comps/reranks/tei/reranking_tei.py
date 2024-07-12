# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import heapq
import json
import os
import re
import time

import requests
from langsmith import traceable

from comps import (
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import RerankingRequest, RerankingResponse, RerankingResponseData

@register_microservice(
    name="opea_service@reranking_tgi_gaudi",
    service_type=ServiceType.RERANK,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=8000,
    input_datatype=RerankingRequest,
    output_datatype=RerankingResponse,
)
@traceable(run_type="llm")
@register_statistics(names=["opea_service@reranking_tgi_gaudi"])
def reranking(request: RerankingRequest) -> RerankingResponse:
    start = time.time()
    if request.retrieved_docs:
        docs = [doc.text for doc in request.retrieved_docs]
        url = tei_reranking_endpoint + "/rerank"
        data = {"query": request.text, "texts": docs}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response_data = response.json()
        best_response_list = heapq.nlargest(request.top_n, response_data, key=lambda x: x["score"])

        ranking_results = []
        for best_response in best_response_list:
            ranking_results.append(RerankingResponseData(
                text=request.retrieved_docs[best_response["index"]].text,
                score=best_response["score"]))
        statistics_dict["opea_service@reranking_tgi_gaudi"].append_latency(time.time() - start, None)
        return RerankingResponse(reranked_docs=ranking_results)
    else:
        return RerankingResponse(reranked_docs=[])


if __name__ == "__main__":
    tei_reranking_endpoint = os.getenv("TEI_RERANKING_ENDPOINT", "http://localhost:8080")
    opea_microservices["opea_service@reranking_tgi_gaudi"].start()
