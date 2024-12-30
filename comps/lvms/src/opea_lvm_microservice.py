# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from fastapi.responses import StreamingResponse
from integrations.opea_llama_vision_lvm import OpeaLlamaVisionLvm
from integrations.opea_llava_lvm import OpeaLlavaLvm
from integrations.opea_predictionguard_lvm import OpeaPredictionguardLvm
from integrations.opea_tgi_llava_lvm import OpeaTgiLlavaLvm
from integrations.opea_video_llama_lvm import OpeaVideoLlamaLvm

from comps import (
    CustomLogger,
    LVMDoc,
    LVMSearchedMultimodalDoc,
    LVMVideoDoc,
    MetadataTextDoc,
    OpeaComponentController,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("opea_lvm_microservice")
logflag = os.getenv("LOGFLAG", False)

# Initialize OpeaComponentController
controller = OpeaComponentController()

# Register components
try:
    # Instantiate LVM components
    if os.getenv("LLAVA_LVM_ENDPOINT"):
        opea_llava_lvm = OpeaLlavaLvm(
            name="OpeaLlavaLvm",
            description="OPEA LLaVA LVM Service",
        )
        controller.register(opea_llava_lvm)

    elif os.getenv("TGI_LLAVA_LVM_ENDPOINT"):
        opea_tgi_llava_lvm = OpeaTgiLlavaLvm(
            name="OpeaTgiLlavaLvm",
            description="OPEA TGI LLaVA LVM Service",
        )
        controller.register(opea_tgi_llava_lvm)

    elif os.getenv("LLAMA_VISION_LVM_ENDPOINT"):
        opea_llama_vision_lvm = OpeaLlamaVisionLvm(
            name="OpeaLlamaVisionLvm",
            description="OPEA LLaMA Vison LVM Service",
        )
        controller.register(opea_llama_vision_lvm)

    elif os.getenv("PREDICTIONGUARD_LVM_ENDPOINT"):
        opea_predictionguard_lvm = OpeaPredictionguardLvm(
            name="OpeaPredictionguardLvm",
            description="OPEA PredictionGuard LVM Service",
        )
        controller.register(opea_predictionguard_lvm)

    elif os.getenv("VIDEO_LLAMA_LVM_ENDPOINT"):
        opea_video_llama_lvm = OpeaVideoLlamaLvm(
            name="OpeaVideoLlamaLvm",
            description="OPEA Video LLaMA LVM Service",
        )
        controller.register(opea_video_llama_lvm)

    # Discover and activate a healthy component
    controller.discover_and_activate()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")


async def stream_forwarder(response, time_start):
    """Forward the stream chunks to the client using iter_content."""
    first_token_latency = None
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            if not first_token_latency:
                first_token_latency = time.time() - time_start
            yield chunk
    statistics_dict["opea_service@lvm"].append_latency(time.time() - time_start, first_token_latency)


@register_microservice(
    name="opea_service@lvm",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9399,
)
@register_statistics(names=["opea_service@lvm"])
async def lvm(
    request: Union[LVMDoc, LVMSearchedMultimodalDoc, LVMVideoDoc]
) -> Union[TextDoc, MetadataTextDoc]:   # can also return a StreamingResponse but omit it in annotation for FastAPI
    start = time.time()

    try:
        # Use the controller to invoke the active component
        lvm_response = await controller.invoke(request)
        if logflag:
            logger.info(lvm_response)

        if controller.active_component.name in ["OpeaVideoLlamaLvm"] or (
            controller.active_component.name in ["OpeaTgiLlavaLvm"] and request.streaming
        ):
            return StreamingResponse(stream_forwarder(lvm_response, start))
        statistics_dict["opea_service@lvm"].append_latency(time.time() - start, None)
        return lvm_response

    except Exception as e:
        logger.error(f"Error during lvm invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA LVM Microservice is starting....")
    opea_microservices["opea_service@lvm"].start()
