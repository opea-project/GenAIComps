# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from integrations.opea_google_search import OpeaGoogleSearch

from comps import (
    EmbedDoc,
    CustomLogger,
    SearchedDoc,
    OpeaComponentController,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("opea_web_retriever_microservice")
logflag = os.getenv("LOGFLAG", False)

# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    # Instantiate google search retriever component
    opea_google_search = OpeaGoogleSearch(
        name="OpeaGoogleSearch",
        description="OPEA Google Search Service",
    )

    # Register components with the controller
    controller.register(opea_google_search)

    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@web_retriever",
    service_type=ServiceType.WEB_RETRIEVER,
    endpoint="/v1/web_retrieval",
    host="0.0.0.0",
    port=7077,
    input_datatype=EmbedDoc,
    output_datatype=SearchedDoc,
)
@register_statistics(names=["opea_service@web_retriever", "opea_service@search"])
async def web_retriever(input: EmbedDoc) -> SearchedDoc:
    start = time.time()

    try:
        # Use the controller to invoke the active component
        res = await controller.invoke(input)
        if logflag:
            logger.info(res)
        statistics_dict["opea_service@web_retriever"].append_latency(time.time() - start, None)
        return res

    except Exception as e:
        logger.error(f"Error during web retriever invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA Web Retriever Microservice is starting....")
    opea_microservices["opea_service@web_retriever"].start()
