# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Union

import requests
from fastapi.responses import StreamingResponse

from comps import CustomLogger, LVMVideoDoc, OpeaComponent, ServiceType

logger = CustomLogger("opea_video_llama_lvm")
logflag = os.getenv("LOGFLAG", False)


class OpeaVideoLlamaLvm(OpeaComponent):
    """A specialized LVM component derived from OpeaComponent for Video-LLaMA services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LVM.name.lower(), description, config)
        self.base_url = os.getenv("LVM_ENDPOINT", "http://localhost:9099")

    async def invoke(
        self,
        request: Union[LVMVideoDoc],
    ) -> Union[StreamingResponse]:
        """Involve the LVM service to generate answer for the provided input.

        Parameters:
        input (LVMVideoDoc): The input containing the video URL, start time, duration, prompt, and maximum new tokens.

        Returns:
        StreamingResponse: A streaming response containing the generated text in text/event-stream format, or a JSON error response if the upstream API responds with an error.
        """
        if logflag:
            logger.info("[lvm] Received input")
            logger.info(request)

        video_url = input.video_url
        chunk_start = input.chunk_start
        chunk_duration = input.chunk_duration
        prompt = input.prompt
        max_new_tokens = input.max_new_tokens

        params = {
            "video_url": video_url,
            "start": chunk_start,
            "duration": chunk_duration,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        logger.info(f"[lvm] Params: {params}")

        response = requests.post(url=f"{self.base_url}/generate", params=params, proxies={"http": None}, stream=True)
        logger.info(f"[lvm] Response status code: {response.status_code}")
        if response.status_code == 200:

            def streamer():
                yield f"{{'video_url': '{video_url}', 'chunk_start': {chunk_start}, 'chunk_duration': {chunk_duration}}}\n".encode(
                    "utf-8"
                )
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
                    logger.info(f"[lvm - chat_stream] Streaming: {chunk}")
                logger.info("[lvm - chat_stream] stream response finished")

            return StreamingResponse(streamer(), media_type="text/event-stream")
        else:
            logger.error(f"[lvm] Error: {response.text}")
            raise logger(status_code=500, detail="The upstream API responded with an error.")

    def check_health(self) -> bool:
        """Checks the health of the embedding service.

        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            # Handle connection errors, timeouts, etc.
            logger.error(f"Health check failed: {e}")
        return False
