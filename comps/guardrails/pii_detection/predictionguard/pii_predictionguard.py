# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import time
from typing import List, Optional

from docarray import BaseDoc
from fastapi import HTTPException
from predictionguard import PredictionGuard

from comps import ServiceType, opea_microservices, register_microservice, register_statistics, statistics_dict

# Initialize Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIIRequestDoc(BaseDoc):
    prompt: str
    replace: Optional[bool] = False
    replace_method: Optional[str] = "random"


class PIIResponseDoc(BaseDoc):
    detected_pii: Optional[List[dict]] = None
    new_prompt: Optional[str] = None


@register_microservice(
    name="opea_service@pii_predictionguard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/pii",
    host="0.0.0.0",
    port=9080,
    input_datatype=PIIRequestDoc,
    output_datatype=PIIResponseDoc,
)
@register_statistics(names=["opea_service@pii_predictionguard"])
def pii_guard(input: PIIRequestDoc) -> PIIResponseDoc:
    start = time.time()

    client = PredictionGuard()

    prompt = input.prompt
    replace = input.replace
    replace_method = input.replace_method

    # Validate replace_method
    if replace_method not in ["random", "mask"]:
        raise HTTPException(status_code=400, detail="Invalid replace method provided.")

    try:
        result = client.pii.check(prompt=prompt, replace=replace, replace_method=replace_method)
    except Exception as e:
        logger.error(f"Error during PII check: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during PII detection")

    statistics_dict["opea_service@pii_predictionguard"].append_latency(time.time() - start, None)

    response_doc = PIIResponseDoc(detected_pii=[], new_prompt=None)

    if "new_prompt" in result["checks"][0]:
        logger.info("PII replaced in the prompt.")
        response_doc.new_prompt = result["checks"][0]["new_prompt"]
    elif "pii_types_and_positions" in result["checks"][0]:
        try:
            detected_pii = json.loads(result["checks"][0]["pii_types_and_positions"])
            response_doc.detected_pii = detected_pii
        except json.JSONDecodeError:
            logger.info("No PII detected in the prompt.")

    return response_doc


if __name__ == "__main__":
    print("Prediction Guard PII Detection initialized.")
    opea_microservices["opea_service@pii_predictionguard"].start()
