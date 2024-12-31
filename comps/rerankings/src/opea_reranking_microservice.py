# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union



from comps import (
    CustomLogger,
    OpeaComponentController,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import ChatCompletionRequest, RerankingRequest, RerankingResponse
from comps.cores.proto.docarray import LLMParamsDoc, LVMVideoDoc, RerankedDoc, SearchedDoc, SearchedMultimodalDoc

logger = CustomLogger("opea_reranking_microservice")
logflag = os.getenv("LOGFLAG", False)
rerank_type = os.getenv("RERANK_TYPE", False)
controller = OpeaComponentController()

# Register components
try:
    # Instantiate reranking components
    if rerank_type == "fastrag":
        from integrations.opea_fastrag import OpeaFastRAGReranking

        opea_fastrag_reranking = OpeaFastRAGReranking(
            name="OpeaFastRAGReranking",
            description="OPEA Fast Rag Service",
        )
        # Register components with the controller
        controller.register(opea_fastrag_reranking)
    elif rerank_type == "tei":
        from integrations.opea_tei import OPEATEIReranking

        opea_tei_reranking = OPEATEIReranking(
            name="OPEATEIReranking",
            description="OPEA TEI Reranking Service",
        )
        # Register components with the controller
        controller.register(opea_tei_reranking)
    elif rerank_type == "videoqna":
        from integrations.opea_videoqna import OPEAVideoQnaReranking

        opea_video_native_reranking = OPEAVideoQnaReranking(
            name="OPEAVideoNativeReranking",
            description="OPEA Video Native Reranking Service",
        )
        # Register components with the controller
        controller.register(opea_video_native_reranking)        

    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@reranking",
    service_type=ServiceType.RERANK,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=8000,
)
@register_statistics(names=["opea_service@reranking"])
async def reranking(
    input: Union[SearchedMultimodalDoc, SearchedDoc, RerankingRequest, ChatCompletionRequest]
) -> Union[RerankedDoc, LLMParamsDoc, RerankingResponse, ChatCompletionRequest, LVMVideoDoc]:
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(f"Input received: {input}")

    try:
        # Use the controller to invoke the active component
        reranking_response = await controller.invoke(input)

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"Output received: {reranking_response}")

        # Record statistics
        statistics_dict["opea_service@reranking"].append_latency(time.time() - start, None)
        return reranking_response

    except Exception as e:
        logger.error(f"Error during reranking invocation: {e}")
        raise


if __name__ == "__main__":
    opea_microservices["opea_service@reranking"].start()
    logger.info("OPEA Reranking Microservice is starting...")
