# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

from predictionguard import PredictionGuard

from comps import (
    ServiceType, 
    TextDoc,
    ScoreDoc, 
    opea_microservices, 
    register_microservice, 
    register_statistics,
    statistics_dict
)


@register_microservice(
    name="opea_service@injection_predictionguard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/injection",
    host="0.0.0.0",
    port=9085,
    input_datatype=TextDoc,
    output_datatype=ScoreDoc
)

@register_statistics(names="opea_service@injection_predictionguard")
def injection_guard(input: TextDoc) -> ScoreDoc:
    client = PredictionGuard()

    text = input.text

    result = client.injection.check(
        prompt=text
    )

    return ScoreDoc(score=result["checks"][0]["probability"])


if __name__ == "__main__":
    print("Prediction Guard Injection Detection initialized.")
    opea_microservices["opea_service@injection_predictionguard"].start()