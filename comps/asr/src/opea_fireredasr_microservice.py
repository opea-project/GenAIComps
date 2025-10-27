#!/usr/bin/env python3

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import List, Union

from fastapi import File, Form, UploadFile, HTTPException, FastAPI
from fastapi.responses import JSONResponse, Response

from comps import (
    Base64ByteStrDoc,
    CustomLogger,
    LLMParamsDoc,
    OpeaComponentLoader,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)
from comps.cores.mega.constants import MCPFuncType
from comps.cores.proto.api_protocol import AudioTranscriptionResponse

logger = CustomLogger("opea_fireredasr_microservice")
logflag = os.getenv("LOGFLAG", False)

# FireRedASR 配置
fireredasr_component_name = os.getenv("FIREREDASR_COMPONENT_NAME", "OPEA_FIREREDASR_ASR")
fireredasr_model_dir = os.getenv("FIREREDASR_MODEL_DIR", "/app/pretrained_models")
fireredasr_asr_type = os.getenv("FIREREDASR_ASR_TYPE", "llm")  # "aed" or "llm"
enable_mcp = os.getenv("ENABLE_MCP", "").strip().lower() in {"true", "1", "yes"}

# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    fireredasr_component_name,
    description=f"OPEA FireRedASR Component: {fireredasr_asr_type}"
)

# Create FastAPI app
app = FastAPI(title="FireRedASR API", version="1.0.0")


@register_microservice(
    name="opea_service@fireredasr_asr",
    service_type=ServiceType.ASR,
    endpoint="/v1/audio/transcriptions",
    host="0.0.0.0",
    port=9099,
    input_datatype=Base64ByteStrDoc,
    output_datatype=LLMParamsDoc,
    enable_mcp=enable_mcp,
    mcp_func_type=MCPFuncType.TOOL,
    description="Convert audio to text using FireRedASR model.",
)
@register_statistics(names=["opea_service@fireredasr_asr"])
async def audio_to_text(
    file: Union[str, UploadFile] = File(...),
    model: str = Form("fireredasr"),
    language: str = Form("auto"),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: List[str] = Form(None),
) -> AudioTranscriptionResponse:
    """
    Convert audio to text using FireRedASR model.
    
    This endpoint follows the OpenAI API specification for audio transcription.
    """
    start = time.time()

    if logflag:
        logger.info("FireRedASR file uploaded.")

    try:
        # Use the loader to invoke the component
        asr_response = await loader.invoke(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            fireredasr_model_dir=fireredasr_model_dir,
            fireredasr_asr_type=fireredasr_asr_type,
        )
        
        if logflag:
            logger.info(asr_response)
        
        statistics_dict["opea_service@fireredasr_asr"].append_latency(time.time() - start, None)
        return asr_response

    except Exception as e:
        logger.error(f"Error during FireRedASR invocation: {e}")
        raise HTTPException(status_code=500, detail=f"FireRedASR processing failed: {str(e)}")


@app.get("/health")
async def health_check() -> Response:
    """Health check endpoint."""
    try:
        # Check if the FireRedASR component is healthy
        fireredasr_component = loader.get_component(fireredasr_component_name)
        if fireredasr_component and fireredasr_component.check_health():
            return Response(status_code=200)
        else:
            return Response(status_code=503, content="Service Unhealthy")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(status_code=503, content="Service Unhealthy")


if __name__ == "__main__":
    logger.info("OPEA FireRedASR Microservice is starting....")
    opea_microservices["opea_service@fireredasr_asr"].start()