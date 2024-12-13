# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from integrations.opea_embedding import OpeaEmbedding
from integrations.predictionguard_embedding import PredictionguardEmbedding

from comps import (
    CustomLogger,
    EmbedDoc,
    OpeaComponentController,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.proto.api_protocol import (
    ChatCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
)

logger = CustomLogger("opea_embedding_microservice")
logflag = os.getenv("LOGFLAG", False)

# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    # Instantiate OpeaEmbedding and PredictionguardEmbedding components
    opea_embedding = OpeaEmbedding(
        name="OpeaEmbedding",
        description="OPEA Embedding Service",
    )
    predictionguard_embedding = PredictionguardEmbedding(
        name="PredictionGuardEmbedding",
        description="Prediction Guard Embedding Service",
    )

    # Register components with the controller
    controller.register(opea_embedding)
    controller.register(predictionguard_embedding)

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
async def embedding(
    input: Union[TextDoc, EmbeddingRequest, ChatCompletionRequest]
) -> Union[EmbedDoc, EmbeddingResponse, ChatCompletionRequest]:
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(f"Input received: {input}")

    try:
        # Use the controller to invoke the active component
        response = controller.invoke(input)

        # Handle response based on input type
        if isinstance(input, TextDoc):
            res = EmbedDoc(text=input.text, embedding=response)
        elif isinstance(input, EmbeddingRequest):
            res = EmbeddingResponse(
                data=[EmbeddingResponseData(index=i, embedding=vec) for i, vec in enumerate(response)]
            )
        elif isinstance(input, ChatCompletionRequest):
            input.embedding = response
            res = input
        else:
            raise TypeError("Unsupported input type")

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"Output generated: {res}")

        # Record statistics
        statistics_dict["opea_service@embedding"].append_latency(time.time() - start, None)
        return res

    except Exception as e:
        logger.error(f"Error during embedding invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA Embedding Microservice is starting...")
    opea_microservices["opea_service@embedding"].start()
