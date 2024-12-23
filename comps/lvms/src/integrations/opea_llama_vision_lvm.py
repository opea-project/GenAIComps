# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import time

import requests
from typing import Union

from comps import CustomLogger, OpeaComponent, ServiceType, LVMDoc, TextDoc

logger = CustomLogger("opea_llama_vision_lvm")
logflag = os.getenv("LOGFLAG", False)


class OpeaLlamaVisionLvm(OpeaComponent):
    """A specialized LVM component derived from OpeaComponent for LLaMA-Vision services."""

    def __init__(self, name: str, description: str, config: dict = None):
        super().__init__(name, ServiceType.LVM.name.lower(), description, config)
        self.base_url = os.getenv("LVM_ENDPOINT", "http://localhost:9399")

    async def invoke(
        self,
        request: Union[LVMDoc],
    ) -> Union[TextDoc]:
        """Involve the LVM service to generate answer for the provided input."""
        if logflag:
            logger.info(request)

        inputs = {"image": request.image, "prompt": request.prompt, "max_new_tokens": request.max_new_tokens}
        # forward to the LLaMA Vision server
        response = requests.post(url=f"{self.base_url}/v1/lvm", data=json.dumps(inputs), proxies={"http": None})

        result = response.json()["text"]
        if logflag:
            logger.info(result)

        return TextDoc(text=result)

    def check_health(self, retries=3, interval=10, timeout=5) -> bool:
        """Checks the health of the LVM service.
        Returns:
            bool: True if the service is reachable and healthy, False otherwise.
        """
        while retries > 0:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=timeout)
                # If status is 200, the service is considered alive
                if response.status_code == 200:
                    return True
            except requests.RequestException as e:
                # Handle connection errors, timeouts, etc.
                print(f"Health check failed: {e}")
            retries -= 1
            time.sleep(interval)
        return False