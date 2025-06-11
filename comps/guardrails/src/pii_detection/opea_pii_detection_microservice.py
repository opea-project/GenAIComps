# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    PIIRequestDoc,
    PIIResponseDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("opea_pii_detection_microservice")
logflag = os.getenv("LOGFLAG", False)

pii_detection_port = int(os.getenv("PII_DETECTION_PORT", 9080))
pii_detection_component_name = os.getenv("PII_DETECTION_COMPONENT_NAME", "OPEA_NATIVE_PII")

if pii_detection_component_name == "OPEA_NATIVE_PII":
    from integrations.piidetection import OpeaPiiDetectionNative
elif pii_detection_component_name == "PREDICTIONGUARD_PII_DETECTION":
    from integrations.predictionguard import OpeaPiiDetectionPredictionGuard
else:
    logger.error(f"Component name {pii_detection_component_name} is not recognized")
    exit(1)


# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    pii_detection_component_name,
    name=pii_detection_component_name,
    description=f"OPEA PII Detection Component: {pii_detection_component_name}",
)


@register_microservice(
    name="opea_service@pii_detection",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/pii",
    host="0.0.0.0",
    port=pii_detection_port,
    input_datatype=Union[TextDoc, PIIRequestDoc],
    output_datatype=Union[TextDoc, PIIResponseDoc],
)
@register_statistics(names=["opea_service@pii_detection"])
async def pii_guard(input: Union[TextDoc, PIIRequestDoc]) -> Union[TextDoc, PIIResponseDoc]:
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(f"Input received: {input}")

    try:
        # Use the loader to invoke the component
        pii_response = await loader.invoke(input)

        # Log the result if logging is enabled
        if logflag:
            logger.info(f"Output received: {pii_response}")

        # Record statistics
        statistics_dict["opea_service@pii_detection"].append_latency(time.time() - start, None)
        return pii_response

    except Exception as e:
        logger.error(f"Error during PII detection invocation: {e}")
        raise


if __name__ == "__main__":
    opea_microservices["opea_service@pii_detection"].start()
    logger.info("OPEA PII Detection Microservice is up and running successfully...")
