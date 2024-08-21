# Copyright (C) 2024 Prediction Guard, Inc.
# SPDX-License-Identified: Apache-2.0

import time
from docarray import BaseDoc
from predictionguard import PredictionGuard
from fastapi import FastAPI, HTTPException

from comps import ServiceType, opea_microservices, register_microservice, register_statistics, statistics_dict


class FactualityDoc(BaseDoc):
    reference: str
    text: str


class ScoreDoc(BaseDoc):
    score: float


@register_microservice(
    name="opea_service@factuality_predictionguard",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/factuality",
    host="0.0.0.0",
    port=9075,
    input_datatype=FactualityDoc,
    output_datatype=ScoreDoc,
)
@register_statistics(names=["opea_service@factuality_predictionguard"])
def factuality_guard(input: FactualityDoc) -> ScoreDoc:
    start = time.time()

    if not input.reference.strip() or not input.text.strip():
        raise HTTPException(status_code=400, detail="Reference and text cannot be empty")

    client = PredictionGuard()

    reference = input.reference
    text = input.text

    result = client.factuality.check(reference=reference, text=text)

    statistics_dict["opea_service@factuality_predictionguard"].append_latency(time.time() - start, None)
    return ScoreDoc(score=result["checks"][0]["score"])


if __name__ == "__main__":
    print("Prediction Guard Factuality initialized.")
    opea_microservices["opea_service@factuality_predictionguard"].start()