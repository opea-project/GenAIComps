# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


# import json
import logging
import os
# import time

from fastapi.responses import StreamingResponse
import requests

from comps import (
    LVMVideoDoc,
    ServiceType,
    opea_microservices,
    register_microservice,
    register_statistics,
)


logging.basicConfig(level=logging.INFO)

@register_microservice(
    name="opea_service@lvm",
    service_type=ServiceType.LVM,
    endpoint="/v1/lvm",
    host="0.0.0.0",
    port=9000,
    input_datatype=LVMVideoDoc,
    output_datatype=StreamingResponse,
)
@register_statistics(names=["opea_service@lvm"])
async def lvm(request: LVMVideoDoc):
    """
    This function handles the LVM microservice, which generates text based on a video URL, start time, duration, prompt, and maximum new tokens.

    Parameters:
    request (LVMVideoDoc): The input request containing the video URL, start time, duration, prompt, and maximum new tokens.

    Returns:
    StreamingResponse: A streaming response containing the generated text in text/event-stream format, or a JSON error response if the upstream API responds with an error.
    """
    logging.info("[lvm] Received request")

    video_url = request.video_url
    chunck_start = request.chunck_start
    chunck_duration = request.chunck_duration
    prompt = request.prompt
    max_new_tokens = request.max_new_tokens # TODO: implement this in the video-llama server

    params = {'video_url': video_url,'start': chunck_start,'duration': chunck_duration,'prompt': prompt,'max_new_tokens': max_new_tokens}
    logging.info(f"[lvm] Params: {params}")

    response = requests.post(url=f"{lvm_endpoint}/generate", params=params, proxies={"http": None}, stream=True)
    logging.info(f"[lvm] Response status code: {response.status_code}")
    if response.status_code == 200:
        def streamer():
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
                logging.info(f"[llm - chat_stream] Streaming: {chunk}")
            logging.info(f"[llm - chat_stream] stream response finished")
        return StreamingResponse(streamer(), media_type="text/event-stream")
    else:
        logging.error(f"[lvm] Error: {response.text}")
        return {"error": "The upstream API responded with an error."}


if __name__ == "__main__":
    lvm_endpoint = os.getenv("LVM_ENDPOINT")

    opea_microservices["opea_service@lvm"].start()
