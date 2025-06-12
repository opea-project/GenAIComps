# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2013--2023, librosa development team.
# Copyright 1999-2003 The OpenLDAP Foundation, Redwood City, California, USA.  All Rights Reserved.
# Copyright (c) 2012, Anaconda, Inc. All rights reserved.

import json
import os
import time

# GenAIComps
from comps import CustomLogger, OpeaComponentLoader
from comps.animation.src.integrations.wav2lip import OpeaAnimation
from comps.cores.mega.constants import MCPFuncType

logger = CustomLogger("opea_animation")
logflag = os.getenv("LOGFLAG", False)
from comps import (
    Base64ByteStrDoc,
    ServiceType,
    VideoPath,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

animation_component_name = os.getenv("ANIMATION_COMPONENT_NAME", "OPEA_ANIMATION")
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}

# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    animation_component_name,
    description=f"OPEA ANIMATION Component: {animation_component_name}",
)


# Register the microservice
@register_microservice(
    name="opea_service@animation",
    service_type=ServiceType.ANIMATION,
    endpoint="/v1/animation",
    host="0.0.0.0",
    port=9066,
    input_datatype=Base64ByteStrDoc,
    output_datatype=VideoPath,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="This function takes an audio piece and a low-quality face image/video as input, fuses mel-spectrogram from the audio with frame(s) from the image/video, and generates a high-quality video of the face with lip movements synchronized with the audio",
)
@register_statistics(names=["opea_service@animation"])
async def animate(audio: Base64ByteStrDoc):
    start = time.time()

    outfile = await loader.invoke(audio.byte_str)
    if logflag:
        logger.info(f"Video generated successfully, check {outfile} for the result.")

    statistics_dict["opea_service@animation"].append_latency(time.time() - start, None)
    return VideoPath(video_path=outfile)


if __name__ == "__main__":
    logger.info("[animation - router] Animation initialized.")
    opea_microservices["opea_service@animation"].start()
