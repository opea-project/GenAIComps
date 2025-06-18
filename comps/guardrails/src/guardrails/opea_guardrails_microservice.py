# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import time
from typing import Union

from dotenv import dotenv_values
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from integrations.llamaguard import OpeaGuardrailsLlamaGuard
from integrations.wildguard import OpeaGuardrailsWildGuard
from pydantic import ValidationError
from utils.llm_guard_input_guardrail import OPEALLMGuardInputGuardrail
from utils.llm_guard_output_guardrail import OPEALLMGuardOutputGuardrail

from comps import (
    CustomLogger,
    GeneratedDoc,
    LLMParamsDoc,
    OpeaComponentLoader,
    SearchedDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("opea_guardrails_microservice")
logflag = os.getenv("LOGFLAG", False)

input_usvc_config = {**dotenv_values("utils/.input_env"), **os.environ}

output_usvc_config = {**dotenv_values("utils/.output_env"), **os.environ}

guardrails_component_name = os.getenv("GUARDRAILS_COMPONENT_NAME", "OPEA_LLAMA_GUARD")
# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    guardrails_component_name,
    name=guardrails_component_name,
    description=f"OPEA Guardrails Component: {guardrails_component_name}",
)

input_guardrail = OPEALLMGuardInputGuardrail(input_usvc_config)
output_guardrail = OPEALLMGuardOutputGuardrail(output_usvc_config)


@register_microservice(
    name="opea_service@guardrails",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/guardrails",
    host="0.0.0.0",
    port=9090,
    input_datatype=Union[LLMParamsDoc, GeneratedDoc, TextDoc],
    output_datatype=Union[TextDoc, GeneratedDoc],
)
@register_statistics(names=["opea_service@guardrails"])
async def safety_guard(input: Union[LLMParamsDoc, GeneratedDoc, TextDoc]) -> Union[TextDoc, GeneratedDoc]:
    start = time.time()

    if logflag:
        logger.info(f"Received input: {input}")

    try:
        if isinstance(input, LLMParamsDoc):
            processed = input_guardrail.scan_llm_input(input)
            if logflag:
                logger.info(f"Input guard passed: {processed}")

        elif isinstance(input, GeneratedDoc):
            try:
                doc = input
            except Exception as e:
                logger.error(f"Problem using input as GeneratedDoc: {e}")
                raise HTTPException(status_code=500, detail=f"{e}") from e
            scanned_output = output_guardrail.scan_llm_output(doc)

            processed = scanned_output
        else:
            processed = input

        # Use the loader to invoke the component
        guardrails_response = await loader.invoke(processed)

        # Record statistics
        statistics_dict["opea_service@guardrails"].append_latency(time.time() - start, None)
        return guardrails_response

    except Exception as e:
        logger.error(f"Error during guardrails invocation: {e}")
        raise


if __name__ == "__main__":
    opea_microservices["opea_service@guardrails"].start()
    logger.info("OPEA guardrails Microservice is up and running successfully...")
