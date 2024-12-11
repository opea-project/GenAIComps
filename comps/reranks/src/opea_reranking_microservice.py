# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from typing import Union

from comps import (
    CustomLogger,
    LLMParamsDoc,
    SearchedDoc,
    ServiceType,
    OpeaComponentController,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    RerankingRequest,
    RerankingResponse,
    RerankingResponseData,
)

from integrations.opea_reranking import OpeaReranking
from integrations.fastrag_reranking import FastragReranking
from integrations.mosec_reranking import MosecReranking
from integrations.videoqna_reranking import VideoqnaReranking

logger = CustomLogger("opea_reranking_microservice")
logflag = os.getenv("LOGFLAG", False)

# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    # Instantiate reranking components
    opea_reranking = OpeaReranking(
        name="OpeaReranking",
        description="OPEA Reranking Service",
    )
    fastrag_reranking = FastragReranking(
        name="FastragReranking",
        description="fastRAG Reranking Service",
    )
    mosec_reranking = MosecReranking(
        name="MosecReranking",
        description="Mosec Reranking Service",
    )
    videoqna_reranking = VideoqnaReranking(
        name="VideoqnaReranking",
        description="VideoQnA Reranking Service",
    )

    # Register components with the controller
    controller.register(opea_reranking)
    controller.register(fastrag_reranking)
    controller.register(mosec_reranking)
    controller.register(videoqna_reranking)

    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")

@register_microservice(
    name="opea_service@reranking",
    service_type=ServiceType.RERANKING,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=8000,
)
@register_statistics(names=["opea_service@reranking"])
async def reranking(
    input: Union[SearchedDoc, RerankingRequest, ChatCompletionRequest]
) -> Union[LLMParamsDoc, RerankingResponse, ChatCompletionRequest]:
    if logflag:
        logger.info(input)
    start = time.time()
    reranking_results = []

    try:
        # Use the controller to invoke the active component
        response = controller.invoke(input)

        if input.retrieved_docs:
            docs = [doc.text for doc in input.retrieved_docs]
            url = tei_reranking_endpoint + "/rerank"
            if isinstance(input, SearchedDoc):
                query = input.initial_query
            else:
                # for RerankingRequest, ChatCompletionRequest
                query = input.input
            data = {"query": query, "texts": docs}
            headers = {"Content-Type": "application/json"}

            for best_response in response_data[: input.top_n]:
                reranking_results.append(
                    {"text": input.retrieved_docs[best_response["index"]].text, "score": best_response["score"]}
                )

        statistics_dict["opea_service@reranking"].append_latency(time.time() - start, None)
        if isinstance(input, SearchedDoc):
            result = [doc["text"] for doc in reranking_results]
            if logflag:
                logger.info(result)
            return LLMParamsDoc(query=input.initial_query, documents=result)
        else:
            reranking_docs = []
            for doc in reranking_results:
                reranking_docs.append(RerankingResponseData(text=doc["text"], score=doc["score"]))
            if isinstance(input, RerankingRequest):
                result = RerankingResponse(reranked_docs=reranking_docs)
                if logflag:
                    logger.info(result)
                return result

            if isinstance(input, ChatCompletionRequest):
                input.reranked_docs = reranking_docs
                input.documents = [doc["text"] for doc in reranking_results]
                if logflag:
                    logger.info(input)
                return input

    except Exception as e:
        logger.error(f"Error during reranking invocation: {e}")
        raise

if __name__ == "__main__":
    logger.info("OPEA Reranking Microservice is starting...")
    opea_microservices["opea_service@reranking"].start()