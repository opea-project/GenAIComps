# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from fastapi import HTTPException

from comps import (
    CustomLogger,
    GeneratedDoc,
    OpeaComponentLoader,
    PromptTemplateInput,
    ServiceType,
    TranslationInput,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.language_detection.src.integrations.native import OPEALanguageDetector

LANG_DETECT_STANDALONE = os.getenv("LANG_DETECT_STANDALONE", "False") == "True"

# Initialize the logger for the microservice
logger = CustomLogger("opea_language_detection")

component_loader = None


# Register the microservice with the specified configuration.
@register_microservice(
    name="opea_service@language_detection",
    service_type=ServiceType.LANGUAGE_DETECTION,
    endpoint="/v1/language_detection",
    host="0.0.0.0",
    port=8069,
    input_datatype=Union[GeneratedDoc, TranslationInput],
    output_datatype=PromptTemplateInput,
)
@register_statistics(names=["opea_service@language_detection"])
# Define a function to handle processing of input for the microservice.
# Its input and output data types must comply with the registered ones above.
async def process(input: Union[GeneratedDoc, TranslationInput]) -> PromptTemplateInput:
    """Process the input document using the OPEALanguageDetector.

    Args:
        input (Union[GeneratedDoc, TranslationInput]): The input document to be processed.

    Returns:
        PromptTemplateInput: The prompt template and placeholders for translation.
    """
    start = time.time()
    try:
        # Pass the input to the 'run' method of the microservice instance
        res = await component_loader.invoke(input)
    except ValueError as e:
        logger.exception(f"An internal error occurred while processing: {str(e)}")
        raise HTTPException(status_code=400, detail=f"An internal error occurred while processing: {str(e)}")
    except Exception as e:
        logger.exception(f"An error occurred while processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing: {str(e)}")
    statistics_dict["opea_service@language_detection"].append_latency(time.time() - start, None)
    return res


if __name__ == "__main__":
    language_detection_component_name = os.getenv("LANGUAGE_DETECTION_COMPONENT_NAME", "OPEA_LANGUAGE_DETECTION")
    # Register components
    try:
        # Initialize OpeaComponentLoader
        component_loader = OpeaComponentLoader(
            language_detection_component_name,
            description=f"OPEA LANGUAGE_DETECTION Component: {language_detection_component_name}",
            config={"is_standalone": LANG_DETECT_STANDALONE},
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        exit(1)

    logger.info("Language detection server started.")
    opea_microservices["opea_service@language_detection"].start()
