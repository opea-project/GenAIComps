# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0


import time
import json
from typing import Optional, List

from docarray import BaseDoc
from predictionguard import PredictionGuard

from comps import (
    ServiceType, 
    opea_microservices, 
    register_microservice, 
    register_statistics,
    statistics_dict
)


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
    output_datatype=PIIResponseDoc
)

@register_statistics(names=["opea_service@pii_predictionguard"])
def pii_guard(input: PIIRequestDoc) -> PIIResponseDoc:
    start = time.time()
    
    client = PredictionGuard()

    prompt = input.prompt
    replace = input.replace
    replace_method = input.replace_method

    result = client.pii.check(
        prompt=prompt,
        replace=replace,
        replace_method=replace_method
    )

    statistics_dict["opea_service@pii_predictionguard"].append_latency(time.time() - start, None)
    if "new_prompt" in result["checks"][0].keys():
        return PIIResponseDoc(new_prompt=result["checks"][0]["new_prompt"])
    elif "pii_types_and_positions" in result["checks"][0].keys():
        detected_pii = json.loads(result["checks"][0]["pii_types_and_positions"])
        return PIIResponseDoc(detected_pii=detected_pii)


if __name__ == "__main__":
    print("Prediction Guard PII Detection initialized.")
    opea_microservices["opea_service@pii_predictionguard"].start()