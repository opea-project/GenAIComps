# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from typing import Union

from comps import (
    CustomLogger,
    #LLMParamsDoc,
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
#from integrations.fastrag_reranking import FastragReranking
#from integrations.mosec_reranking import MosecReranking
#from integrations.videoqna_reranking import VideoqnaReranking


# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    # Instantiate reranking components
    opea_reranking = OpeaReranking(
        name="OpeaReranking",
        description="OPEA Reranking Service",
    )

    # Register components with the controller
    controller.register(opea_reranking)

    # Discover and activate a healthy component
    controller.discover_and_activate()

    # prepare input
    #response = await controller.invoke(input)

    import asyncio
    asyncio.run(controller.invoke(input))
    #print(response)

except Exception as e:
    print(f"Failed to initialize components: {e}")

