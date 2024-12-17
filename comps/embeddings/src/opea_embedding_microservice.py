# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from comps import (
    CustomLogger,
    OpeaComponentController,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from integrations.opea_tei_embedding import OpeaTEIEmbedding
from integrations.predictionguard_embedding import PredictionguardEmbedding
from integrations.opea_multimodal_embedding_bridgetower import  OpeaMultimodalEmbeddingBrigeTower
from integrations.opea_mosec_embedding import OpeaMosecEmbedding

logger = CustomLogger("opea_embedding_microservice")
logflag = os.getenv("LOGFLAG", False)

# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    # Instantiate Embedding components
    opea_tei_embedding = OpeaTEIEmbedding(
        name="OpeaTEIEmbedding",
        description="OPEA TEI Embedding Service",
    )
    opea_mosec_embedding = OpeaMosecEmbedding(
        name="OpeaMosecEmbedding",
        description="OPEA MOSEC Embedding Service",
    )
    predictionguard_embedding = PredictionguardEmbedding(
        name="PredictionGuardEmbedding",
        description="Prediction Guard Embedding Service",
    )
    bridgetower_embedding = OpeaMultimodalEmbeddingBrigeTower(
        name="OpeaMultimodalEmbeddingBrigeTower",
        description="OPEA BredgeTower Multimodal Embedding Service",
    )

    # Register components with the controller
    controller.register(opea_tei_embedding)
    controller.register(opea_mosec_embedding)
    controller.register(predictionguard_embedding)
    controller.register(bridgetower_embedding)

    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@embedding",
    service_type=ServiceType.EMBEDDING,
    endpoint="/v1/embeddings",
    host="0.0.0.0",
    port=6000,
)
@register_statistics(names=["opea_service@embedding"])
async def embedding(input: EmbeddingRequest) -> EmbeddingResponse:
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(f"Input received: {input}")

    try:
        # Use the controller to invoke the active component
        embedding_response = controller.invoke(input)

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
    logger.info("OPEA Embedding Microservice is starting...")
    opea_microservices["opea_service@embedding"].start()
