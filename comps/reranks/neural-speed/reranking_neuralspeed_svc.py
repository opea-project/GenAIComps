# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import time
from typing import List, Optional

import httpx
import msgspec
import requests
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from comps import (
    CustomLogger,
    LLMParamsDoc,
    SearchedDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)


@register_microservice(
    name="opea_service@reranking_mosec",
    service_type=ServiceType.RERANK,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=8000,
    input_datatype=SearchedDoc,
    output_datatype=LLMParamsDoc,
)

@traceable(run_type="reranking")
@register_statistics(names=["opea_service@reranking_mosec"])
def reranking(input: SearchedDoc) -> LLMParamsDoc:
    start = time.time()
    if input.retrieved_docs:
        docs = [doc.text for doc in input.retrieved_docs]
        url = mosec_reranking_endpoint + "/inference"
        data = {"query": input.initial_query, "docs": docs}
        resp = requests.post(url, data=msgspec.msgpack.encode(data))
        response = msgspec.msgpack.decode(resp.content)["scores"]

        best_response_index = max(enumerate(response), key=lambda score: score[1])[0]
        doc = input.retrieved_docs[best_response_index]
        if doc.text and len(re.findall("[\u4E00-\u9FFF]", doc.text)) / len(doc.text) >= 0.3:
            # chinese context
            template = "仅基于以下背景回答问题:\n{context}\n问题: {question}"
        else:
            template = """Answer the question based only on the following context:
    {context}
    Question: {question}
            """
        prompt = ChatPromptTemplate.from_template(template)
        final_prompt = prompt.format(context=doc.text, question=input.initial_query)
        statistics_dict["opea_service@reranking_mosec"].append_latency(time.time() - start, None)

        return LLMParamsDoc(query=final_prompt.strip())
    else:
        if logflag:
            logger.info(input.initial_query)
        return LLMParamsDoc(query=input.initial_query)


if __name__ == "__main__":
    mosec_reranking_endpoint = os.getenv("MOSEC_RERANKING_ENDPOINT", "http://localhost:8080")
    print("NeuralSpeed Reranking Microservice Initialized.")
    opea_microservices["opea_service@reranking_mosec"].start()
