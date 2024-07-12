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
    name="opea_service@factuality_predictionguard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/factuality",
    host="0.0.0.0",
    port="9075",
    input_datatype=TextDoc,
    output_datatype=TextDoc
)

@register_statistics(names="opea_service@factuality_predictionguard")
def factuality_guard(input: TextDoc) -> TextDoc:
    client = PredictionGuard()

    reference = input.reference
    text = input.text

    result = client.factuality.check(
        reference=reference,
        text=text
    )

    return TextDoc(text=result["checks"][0]["score"])


if __name__ == "__main__":
    print("Prediction Guard Factuality initialized.")
    opea_microservices["opea_service@factuality_predictionguard"].start()