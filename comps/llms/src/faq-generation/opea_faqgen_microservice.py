# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from integrations.opea import OPEAFAQGen_TGI, OPEAFAQGen_vLLM

from comps import (
    CustomLogger,
    LLMParamsDoc,
    OpeaComponentController,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("llm_faqgen")
logflag = os.getenv("LOGFLAG", False)

llm_backend = os.getenv("LLM_BACKEND", "").lower()
if logflag:
    logger.info(f"LLM BACKEND: {llm_backend}")

comps_name = {"tgi": "OPEAFAQGen_TGI", "vllm": "OPEAFAQGen_vLLM"}
active_comps_name = comps_name[llm_backend] if llm_backend != "" else ""


# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    opea_faqgen_tgi = OPEAFAQGen_TGI(
        name=comps_name["tgi"],
        description="OPEA FAQGen Service",
    )
    # Register components with the controller
    controller.register(opea_faqgen_tgi)

    opea_faqgen_vllm = OPEAFAQGen_vLLM(
        name=comps_name["vllm"],
        description="OPEA FAQGen Service",
    )
    # Register components with the controller
    controller.register(opea_faqgen_vllm)

    # Discover and activate a healthy component
    controller.discover_and_activate(active_comps_name)
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


@register_microservice(
    name="opea_service@llm_faqgen",
    service_type=ServiceType.LLM,
    endpoint="/v1/faqgen",
    host="0.0.0.0",
    port=9000,
)
@register_statistics(names=["opea_service@llm_faqgen"])
async def llm_generate(input: LLMParamsDoc):
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(input)

    try:
        # Use the controller to invoke the active component
        response = await controller.invoke(input)
        # Record statistics
        statistics_dict["opea_service@llm_faqgen"].append_latency(time.time() - start, None)
        return response

    except Exception as e:
        logger.error(f"Error during FaqGen invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA FAQGen Microservice is starting...")
    opea_microservices["opea_service@llm_faqgen"].start()
