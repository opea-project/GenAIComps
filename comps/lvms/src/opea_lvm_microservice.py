# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from integrations.llama_vision import OpeaLlamaVisionLvm
from integrations.llava import OpeaLlavaLvm
from integrations.predictionguard import OpeaPredictionguardLvm
from integrations.tgi_llava import OpeaTgiLlavaLvm
from integrations.video_llama import OpeaVideoLlamaLvm

from comps import (
    CustomLogger,
    LVMDoc,
    LVMSearchedMultimodalDoc,
    LVMVideoDoc,
    MetadataTextDoc,
    OpeaComponentLoader,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("opea_lvm_microservice")
logflag = os.getenv("LOGFLAG", False)

lvm_component_name = os.getenv("LVM_COMPONENT_NAME", "OPEA_LLAVA_LVM")
# Initialize OpeaComponentController
loader = OpeaComponentLoader(lvm_component_name, description=f"OPEA LVM Component: {lvm_component_name}")


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
) -> Union[TextDoc, MetadataTextDoc]:  # can also return a StreamingResponse but omit it in annotation for FastAPI
    start = time.time()

    try:
        # Use the controller to invoke the active component
        lvm_response = await loader.invoke(request)
        if logflag:
            logger.info(lvm_response)

        if loader.component.name in ["OpeaVideoLlamaLvm"] or (
            loader.component.name in ["OpeaTgiLlavaLvm"] and request.streaming
        ):
            # statistics for StreamingResponse are handled inside the integrations
            # here directly return the response
            return lvm_response
        statistics_dict["opea_service@lvm"].append_latency(time.time() - start, None)
        return lvm_response

    except Exception as e:
        logger.error(f"Error during lvm invocation: {e}")
        raise


if __name__ == "__main__":
    logger.info("OPEA LVM Microservice is starting....")
    opea_microservices["opea_service@lvm"].start()
