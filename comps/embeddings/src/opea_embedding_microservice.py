# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from integrations.clip import OpeaClipEmbedding
from integrations.ovms import OpeaOVMSEmbedding
from integrations.predictionguard import PredictionguardEmbedding
from integrations.tei import OpeaTEIEmbedding

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.mega.constants import MCPFuncType
from comps.cores.proto.api_protocol import EmbeddingRequest, EmbeddingResponse
from comps.cores.telemetry.opea_telemetry import opea_telemetry


logger = CustomLogger("opea_embedding_microservice")
logflag = os.getenv("LOGFLAG", False)

embedding_component_name = os.getenv("EMBEDDING_COMPONENT_NAME", "OPEA_TEI_EMBEDDING")
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}
# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    embedding_component_name,
    description=f"OPEA Embedding Component: {embedding_component_name}",
)


@register_microservice(
    name="opea_service@embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Generate embeddings for given texts",
)
@opea_telemetry
@register_statistics(names=["opea_service@embedding"])
async def embedding(input: EmbeddingRequest) -> EmbeddingResponse:
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(f"Input received: {input}")

    try:
        # Use the loader to invoke the component
        embedding_response = await loader.invoke(input)

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"Output received: {embedding_response}")

        # Record statistics
        statistics_dict["opea_service@embedding"].append_latency(time.time() - start, None)
        return embedding_response

    except Exception as e:
        logger.error(f"Error during embedding invocation: {e}")
        raise


if __name__ == "__main__":
    opea_microservices["opea_service@embedding"].start()
    logger.info("OPEA Embedding Microservice is up and running successfully...")
