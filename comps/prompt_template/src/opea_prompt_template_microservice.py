# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time

from fastapi import HTTPException

from comps import (
    CustomLogger,
    LLMParamsDoc,
    OpeaComponentLoader,
    PromptTemplateInput,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.prompt_template.src.integrations.native import OPEAPromptTemplateGenerator

logger = CustomLogger("opea_prompt_template")
component_loader = None


@register_microservice(
    name="opea_service@prompt_template",
    service_type=ServiceType.PROMPT_TEMPLATE,
    endpoint="/v1/prompt_template",
    host="0.0.0.0",
    port=7900,
    input_datatype=PromptTemplateInput,
    output_datatype=LLMParamsDoc,
)
@register_statistics(names=["opea_service@prompt_template"])
async def process(input: PromptTemplateInput) -> LLMParamsDoc:
    """Process the input document using the OPEALanguageDetector.

    Args:
        input (PromptTemplateInput): The input document to be processed.

    Returns:
        LLMParamsDoc: The processed document with LLM parameters.
    """
    start = time.time()
    try:
        res = await component_loader.invoke(input)
    except ValueError as e:
        logger.exception(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Unhandled error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    statistics_dict["opea_service@prompt_template"].append_latency(time.time() - start, None)
    return res


if __name__ == "__main__":
    prompt_templete_component_name = os.getenv("PROMPT_TEMPLATE_COMPONENT_NAME", "OPEA_PROMPT_TEMPLATE")

    try:
        component_loader = OpeaComponentLoader(
            prompt_templete_component_name,
            description=f"Prompt Template Generator Component: {prompt_templete_component_name}",
            config={},
        )
    except Exception as e:
        logger.error(f"Failed to initialize component: {e}")
        exit(1)

    logger.info("Prompt template service started.")
    opea_microservices["opea_service@prompt_template"].start()
