# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import json
import os
import time
from typing import Union
import requests

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
from template import ChatTemplate

logger = CustomLogger("lvm")
logflag = os.getenv("LOGFLAG", False)


@register_microservice(
    name="opea_service@lvm",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9399,
)
@register_statistics(names=["opea_service@lvm"])
async def lvm(request: Union[LVMDoc, SearchedMultimodalDoc]) -> TextDoc:
    if logflag:
        logger.info(request)
    start = time.time()
    if isinstance(request, SearchedMultimodalDoc):
        if logflag:
            logger.info("[SearchedMultimodalDoc ] input from retriever microservice")
        retrieved_metadatas = request.metadata
        if application == "MM_RAG_ON_VIDEOS":
            img_b64_str = retrieved_metadatas[0]['b64_img_str']
            initial_query = request.initial_query
            prompt = ChatTemplate.generate_multimodal_rag_on_videos_prompt(initial_query, retrieved_metadatas)
            # use default lvm parameters for inferencing
            new_input = LVMDoc(image=img_b64_str, prompt=prompt)
            max_new_tokens = new_input.max_new_tokens
            if logflag:
                logger.info(f"prompt generated for [SearchedMultimodalDoc ] input from retriever microservice: {prompt}")
        else:
            raise NotImplementedError(f"For application {application}: it has NOT implemented SearchedMultimodalDoc input from retriever microservice!")

    else:
        img_b64_str = request.image
        prompt = request.prompt
        max_new_tokens = request.max_new_tokens

    inputs = {"img_b64_str": img_b64_str, "prompt": prompt, "max_new_tokens": max_new_tokens}
    # forward to the LLaVA server
    response = requests.post(url=f"{lvm_endpoint}/generate", data=json.dumps(inputs), proxies={"http": None})

    statistics_dict["opea_service@lvm"].append_latency(time.time() - start, None)
    result = response.json()["text"]
    if logflag:
        logger.info(result)
    return TextDoc(text=result)


if __name__ == "__main__":
    lvm_endpoint = os.getenv("LVM_ENDPOINT", "http://localhost:8399")
    application = os.getenv("APPLICATION", None)

    logger.info("[LVM] LVM initialized.")
    opea_microservices["opea_service@lvm"].start()
