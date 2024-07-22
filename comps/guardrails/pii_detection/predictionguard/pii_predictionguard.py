# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0


import time

from predictionguard import PredictionGuard

from comps import (
    ServiceType, 
    TextDoc,
    PIIDoc,
    opea_microservices, 
    register_microservice, 
    register_statistics,
    statistics_dict
)


@register_microservice(
    name="opea_service@pii_predictionguard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/pii",
    host="0.0.0.0",
    port=9080,
    input_datatype=PIIDoc,
    output_datatype=TextDoc
)

@register_statistics(names="opea_service@pii_predictionguard")
def pii_guard(input: PIIDoc) -> TextDoc:
    start = time.time()
    
    client = PredictionGuard()

    prompt = input.prompt
    replace = input.replace
    replace_method = input.replace_method

    result = client.pii.check(
        prompt=prompt,
        replace=replace,
        replace_method = replace_method
    )

    statistics_dict["opea_service@pii_predictionguard"].append_latency(time.time() - start, None)
    if "new_prompt" in result["checks"][0].keys():
        return TextDoc(text=result["checks"][0]["new_prompt"])
    elif "pii_types_and_positions" in result["checks"][0].keys():
        return TextDoc(text=result["checks"][0]["pii_types_and_positions"])


if __name__ == "__main__":
    print("Prediction Guard PII Detection initialized.")
    opea_microservices["opea_service@pii_predictionguard"].start()