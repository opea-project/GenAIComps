# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
import asyncio
from typing import Union
from dotenv import dotenv_values
from fastapi import HTTPException

from utils.llm_guard_input_guardrail import (
    OPEALLMGuardInputGuardrail
)
from utils.llm_guard_output_guardrail import (
    OPEALLMGuardOutputGuardrail
)

from comps import (
    CustomLogger,
    GeneratedDoc,
    LLMParamsDoc,
    SearchedDoc,
    OpeaComponentLoader,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

from comps.cores.proto.api_protocol import ChatCompletionRequest, DocSumChatCompletionRequest

logger = CustomLogger("opea_guardrails_microservice")
logflag = os.getenv("LOGFLAG", False)

usvc_config = {
    **dotenv_values(".env"), 
    **os.environ  
}

guardrails_component_name = os.getenv("GUARDRAILS_COMPONENT_NAME", "OPEA_LLAMA_GUARD")
# Initialize OpeaComponentLoader
loader = OpeaComponentLoader(
    guardrails_component_name,
    name=guardrails_component_name,
    description=f"OPEA Guardrails Component: {guardrails_component_name}",
)

input_guardrail = OPEALLMGuardInputGuardrail(usvc_config)
output_guardrail = OPEALLMGuardOutputGuardrail(usvc_config)

@register_microservice(
    name="opea_service@guardrails",
    service_type=ServiceType.GUARDRAIL,
    endpoint="/v1/guardrails",
    host="0.0.0.0",
    port=9090,
    input_datatype=Union[LLMParamsDoc, GeneratedDoc, ChatCompletionRequest, SearchedDoc, ChatCompletionRequest, DocSumChatCompletionRequest],
    output_datatype=Union[LLMParamsDoc, GeneratedDoc, ChatCompletionRequest, SearchedDoc, ChatCompletionRequest, DocSumChatCompletionRequest],
)
@register_statistics(names=["opea_service@guardrails"])
async def safety_guard(input: Union[LLMParamsDoc, GeneratedDoc, ChatCompletionRequest, SearchedDoc, ChatCompletionRequest, DocSumChatCompletionRequest]) -> Union[LLMParamsDoc, GeneratedDoc, ChatCompletionRequest, SearchedDoc, ChatCompletionRequest, DocSumChatCompletionRequest]:
    start_time = time.time()
    
    if logflag:
        logger.info(f"Received input: {input}")
    
    try:
        if isinstance(input, LLMParamsDoc):
            processed = input_guardrail.scan_llm_input(input)
            
            statistics_dict["opea_service@guardrails"].append_latency(
                time.time() - start_time, 
                f"input_guard:{type(input).__name__}"
            )
            
            if logflag:
                logger.info(f"Input guard passed: {processed}")
            return processed
        
        elif isinstance(input, GeneratedDoc):
            processed = output_guardrail.scan_llm_output(input)
            
            if os.getenv("APPLY_CONTENT_GUARD", "true").lower() == "true":
                text_doc = TextDoc(text=processed.text)
                content_guard_result = await loader.invoke(text_doc)
                processed.text = content_guard_result.text
            
            statistics_dict["opea_service@guardrails"].append_latency(
                time.time() - start_time, 
                f"output_guard:{type(input).__name__}"
            )
            
            if logflag:
                logger.info(f"Output guard passed: {processed}")
            return processed
    
    except HTTPException as e:
        if e.status_code == 466:
            logger.warning(f"Security rejection: {e.detail}")
            statistics_dict["opea_service@guardrails"].append_latency(
                time.time() - start_time, 
                f"rejection:{e.status_code}"
            )
        raise e
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        statistics_dict["opea_service@guardrails"].append_latency(
            time.time() - start_time, 
            f"error:{type(e).__name__}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    opea_microservices["opea_service@guardrails"].start()
    logger.info("OPEA guardrails Microservice is up and running successfully...")
