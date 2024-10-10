# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
import os
import pickle
import threading
from typing import Dict, Tuple

from vllm import LLM, SamplingParams

from comps import CustomLogger, ServiceType, SpecDecodeParams, opea_microservices, opea_telemetry, register_microservice

from .drafter import get_llm

logger = CustomLogger("spec_decode_drafter_vllm")
logflag = os.getenv("LOGFLAG", False)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


# Create an Scorer.
@register_microservice(
    name="opea_service@spec_decode_scorer_vllm",
    service_type=ServiceType.SPEC_DECODE_SCORER,
    endpoint="/v1/spec_decode/scorer/completions",
    host="0.0.0.0",
    port=8016,
)
def llm_generate(input: SpecDecodeParams):
    if logflag:
        logger.info(input)
    # reuse the llm model
    # will use scorer to set the env for drafter
    os.environ["LLM_MODEL"] = input.model
    os.environ["SPEC_MODEL"] = input.speculative_model
    os.environ["NUM_SPECULATIVE_TOKENS"] = input.num_speculative_tokens
    llm = get_llm(spec_step="spec", device=input.device, tensor_parallel_size=input.tensor_parallel_size)

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    response = llm.generate(input.query, sampling_params)
    if logflag:
        logger.info(response.outputs[0].text)
    return GeneratedDoc(text=response.outputs[0].text, prompt=input.query)


if __name__ == "__main__":
    opea_microservices["opea_service@spec_decode_scorer_vllm"].start()
