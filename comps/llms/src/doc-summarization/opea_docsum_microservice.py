# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from integrations.opea import OPEADocSum_TGI, OPEADocSum_vLLM

from comps import (
    CustomLogger,
    DocSumLLMParams,
    OpeaComponentController,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("llm_docsum")
logflag = os.getenv("LOGFLAG", False)

llm_backend = os.getenv("LLM_BACKEND", "").lower()
if logflag:
    logger.info(f"LLM BACKEND: {llm_backend}")

comps_name = {"tgi": "OPEADocSum_TGI", "vllm": "OPEADocSum_vLLM"}
active_comps_name = comps_name[llm_backend] if llm_backend != "" else ""

# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    opea_docsum_tgi = OPEADocSum_TGI(
        name=comps_name["tgi"],
        description="OPEA DocSum Service",
    )
    # Register components with the controller
    controller.register(opea_docsum_tgi)

    opea_docsum_vllm = OPEADocSum_vLLM(
        name=comps_name["vllm"],
        description="OPEA DocSum Service",
    )
    # Register components with the controller
    controller.register(opea_docsum_vllm)

    # Discover and activate a healthy component
    controller.discover_and_activate(active_comps_name)
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@llm_docsum",
    service_type=ServiceType.LLM,
    endpoint="/v1/docsum",
    host="0.0.0.0",
    port=9000,
)
@register_statistics(names=["opea_service@llm_docsum"])
async def llm_generate(input: DocSumLLMParams):
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(input)

    try:
        # Use the controller to invoke the active component
        response = await controller.invoke(input)
        # Record statistics
        statistics_dict["opea_service@llm_docsum"].append_latency(time.time() - start, None)
        return response

    except Exception as e:
        logger.error(f"Error during DocSum invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA DocSum Microservice is starting...")
    opea_microservices["opea_service@llm_docsum"].start()
