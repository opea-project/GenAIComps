# Copyright (C) 2025 Zensar Technologies Private Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import time

from comps import (
    CustomLogger,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.arb_post_hearing_assistant.src.integrations.tgi import OpeaArbPostHearingAssistantTgi
from comps.arb_post_hearing_assistant.src.integrations.vllm import OpeaArbPostHearingAssistantVllm
from comps.cores.proto.api_protocol import ArbPostHearingAssistantChatCompletionRequest

logger = CustomLogger("arb_post_hearing_assistant_microservice")
logflag = os.getenv("LOGFLAG", False)

llm_component_name = os.getenv("OPEA_ARB_POSTHEARING_ASSISTANT_COMPONENT_NAME", "OpeaArbPostHearingAssistantTgi")

# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    llm_component_name, description=f"OPEA LLM arb_posthearing_assistant Component: {llm_component_name}"
)

port = int(os.getenv("OPEA_ARB_POSTHEARING_ASSISTANT_PORT", 9000))


@register_microservice(
    name="opea_service@arb_post_hearing_assistant",
    service_type=ServiceType.ARB_POST_HEARING_ASSISTANT,
    endpoint="/v1/arb-post-hearing",
    host="0.0.0.0",
    port=port,
)
@register_statistics(names=["opea_service@arb_post_hearing_assistant"])
async def llm_generate(input: ArbPostHearingAssistantChatCompletionRequest):
    start = time.time()

    # Log the input if logging is enabled
    if logflag:
        logger.info(input)

    try:
        # Use the controller to invoke the active component
        response = await loader.invoke(input)
        # Record statistics
        statistics_dict["opea_service@arb_post_hearing_assistant"].append_latency(time.time() - start, None)
        return response

    except Exception as e:
        logger.error(f"Error during arb_post_hearing_assistant invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA arb_post_hearing_assistant Microservice is starting...")
    opea_microservices["opea_service@arb_post_hearing_assistant"].start()
