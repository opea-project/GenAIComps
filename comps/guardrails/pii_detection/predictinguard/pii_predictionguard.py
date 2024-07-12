# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

from predictionguard import PredictionGuard

from comps import (
    ServiceType, 
    TextDoc, 
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
    port="9080",
    input_datatype=TextDoc,
    output_datatype=TextDoc
)

@register_statistics(names="opea_service@pii_predictionguard")
def pii_guard(input: TextDoc) -> TextDoc:
    client = PredictionGuard()

    prompt = input.prompt
    replace = input.replace
    replace_method = input.replace_method

    result = client.injection.check(
        prompt=prompt,
        replace=replace,
        replace_method = replace_method
    )

    return TextDoc(text=result["checks"][0]["new_prompt"])


if __name__ == "__main__":
    print("Prediction Guard PII Detection initialized.")
    opea_microservices["opea_service@pii_predictionguard"].start()