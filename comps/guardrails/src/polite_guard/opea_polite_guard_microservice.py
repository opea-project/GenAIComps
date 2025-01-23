# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from integrations.politeguard import OpeaPoliteGuard

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("opea_polite_guard_microservice")
logflag = os.getenv("LOGFLAG", False)

polite_guard_component_name = os.getenv("POLITE_GUARD_COMPONENT_NAME", "OPEA_POLITE_GUARD")
# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    polite_guard_component_name,
    name=polite_guard_component_name,
    description=f"OPEA Polite Guard Component: {polite_guard_component_name}",
)


@register_microservice(
    name="opea_service@polite_guard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/polite",
    host="0.0.0.0",
    port=9092,
    input_datatype=TextDoc,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@polite_guard"])
async def llm_generate(input: TextDoc):
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(f"Input received: {input}")

    try:
        # Use the loader to invoke the component
        bias_response = await loader.invoke(input.text)

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"Output received: {bias_response}")

        # Record statistics
        statistics_dict["opea_service@polite_guard"].append_latency(time.time() - start, None)
        return bias_response

    except Exception as e:
        logger.error(f"Error during polite guard invocation: {e}")
        raise


if __name__ == "__main__":
    opea_microservices["opea_service@polite_guard"].start()
    logger.info("OPEA Polite Guard Microservice is up and running successfully...")
