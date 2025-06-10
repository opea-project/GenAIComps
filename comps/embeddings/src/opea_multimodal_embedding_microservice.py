# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from integrations.multimodal_bridgetower import OpeaMultimodalEmbeddingBrigeTower

from comps import (
    CustomLogger,
    EmbedMultimodalDoc,
    MultimodalDoc,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.mega.constants import MCPFuncType

logger = CustomLogger("opea_multimodal_embedding_microservice")
logflag = os.getenv("LOGFLAG", False)

embedding_component_name = os.getenv("EMBEDDING_COMPONENT_NAME", "OPEA_MULTIMODAL_EMBEDDING_BRIDGETOWER")
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}
# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    embedding_component_name,
    description=f"OPEA Embedding Component: {embedding_component_name}",
)

port = int(os.getenv("MM_EMBEDDING_PORT_MICROSERVICE", 6000))


@register_microservice(
    name="opea_service@multimodal_embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=port,
    input_datatype=MultimodalDoc,
    output_datatype=EmbedMultimodalDoc,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Generate embeddings for multimodal documents",
)
@register_statistics(names=["opea_service@multimodal_embedding"])
async def embedding(input: MultimodalDoc) -> EmbedMultimodalDoc:
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
        statistics_dict["opea_service@multimodal_embedding"].append_latency(time.time() - start, None)
        return embedding_response

    except Exception as e:
        logger.error(f"Error during embedding invocation: {e}")
        raise


if __name__ == "__main__":
    opea_microservices["opea_service@multimodal_embedding"].start()
    logger.info("OPEA Multimodal Embedding Microservice is up and running successfully...")
