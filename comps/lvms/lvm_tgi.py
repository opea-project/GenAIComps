# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import time
from typing import Union

from fastapi.responses import StreamingResponse
from huggingface_hub import AsyncInferenceClient
from template import ChatTemplate

from comps import (
    CustomLogger,
    LVMDoc,
    SearchedMultimodalDoc,
    ServiceType,
    TextDoc,
    opea_microservices,
    register_microservice,
    register_statistics,
    statistics_dict,
)

logger = CustomLogger("lvm_tgi")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@lvm_tgi",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9399,
    input_datatype=LVMDoc,
    output_datatype=TextDoc,
)
@register_statistics(names=["opea_service@lvm_tgi"])
async def lvm(request: Union[LVMDoc, SearchedMultimodalDoc]) -> TextDoc:
    if logflag:
        logger.info(request)
    start = time.time()
    if isinstance(request, SearchedMultimodalDoc):
        # This is to construct LVMDoc from SearchedMultimodalDoc input from retriever microservice
        #  for Multimodal RAG on Videos application
        if logflag:
            logger.info("[SearchedMultimodalDoc ] input from retriever microservice")
        retrieved_metadatas = request.metadata
        if application == "MM_RAG_ON_VIDEOS":
            img_b64_str = retrieved_metadatas[0]["b64_img_str"]
            initial_query = request.initial_query
            prompt = ChatTemplate.generate_multimodal_rag_on_videos_prompt(initial_query, retrieved_metadatas)
            # use default lvm parameters for inferencing
            new_request = LVMDoc(image=img_b64_str, prompt=prompt)
            if logflag:
                logger.info(
                    f"prompt generated for [SearchedMultimodalDoc ] input from retriever microservice: {prompt}"
                )
        else:
            raise NotImplementedError(
                f"For application {application}: it has NOT implemented SearchedMultimodalDoc input from retriever microservice!"
            )
    else:
        new_request = request
    stream_gen_time = []
    img_b64_str = new_request.image
    prompt = new_request.prompt
    max_new_tokens = new_request.max_new_tokens
    streaming = new_request.streaming
    repetition_penalty = new_request.repetition_penalty
    temperature = new_request.temperature
    top_k = new_request.top_k
    top_p = new_request.top_p

    image = f"data:image/png;base64,{img_b64_str}"
    image_prompt = f"![]({image})\n{prompt}\nASSISTANT:"

    if streaming:

        async def stream_generator():
            chat_response = ""
            text_generation = await lvm_client.text_generation(
                prompt=image_prompt,
                stream=streaming,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            async for text in text_generation:
                stream_gen_time.append(time.time() - start)
                chat_response += text
                chunk_repr = repr(text.encode("utf-8"))
                if logflag:
                    logger.info(f"[llm - chat_stream] chunk:{chunk_repr}")
                yield f"data: {chunk_repr}\n\n"
            if logflag:
                logger.info(f"[llm - chat_stream] stream response: {chat_response}")
            statistics_dict["opea_service@lvm_tgi"].append_latency(stream_gen_time[-1], stream_gen_time[0])
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        generated_str = await lvm_client.text_generation(
            image_prompt,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        statistics_dict["opea_service@lvm_tgi"].append_latency(time.time() - start, None)
        if logflag:
            logger.info(generated_str)
        return TextDoc(text=generated_str)


if __name__ == "__main__":
    lvm_endpoint = os.getenv("LVM_ENDPOINT", "http://localhost:8399")
    application = os.getenv("APPLICATION", None)
    lvm_client = AsyncInferenceClient(lvm_endpoint)
    logger.info("[LVM] LVM initialized.")
    opea_microservices["opea_service@lvm_tgi"].start()
