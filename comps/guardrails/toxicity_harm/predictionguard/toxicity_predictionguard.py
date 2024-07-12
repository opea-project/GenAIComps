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
    name="opea_service@toxicity_predictionguard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/guardrails",
    host="0.0.0.0",
    post="9090",
    input_datatype=TextDoc,
    output_datatype=TextDoc
)

@register_statistics(names="opea_service@toxicity_predictionguard")
def toxicity_guard(input: TextDoc) -> TextDoc:
    client = PredictionGuard()

    text = input.text

    result = client.toxicity.check(
        text=text
    )

    return TextDoc(text=result["checks"][0]["score"])


if __name__ == "__main__":
    print("Prediction Guard Toxicity initialized.")
    opea_microservices["opea_service@toxicity_predictionguard"].start()