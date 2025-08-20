#!/bin/bash
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import random
import sys

import requests
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def validate_svc(ip_address, service_port):

    endpoint = f"http://{ip_address}:{service_port}"

    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            result = await session.initialize()
            url = "https://github.com/intel/intel-extension-for-transformers/raw/main/intel_extension_for_transformers/neural_chat/assets/audio/sample.wav"
            response = requests.get(url)
            response.raise_for_status()  # Ensure the download succeeded
            binary_data = response.content
            base64_str = base64.b64encode(binary_data).decode("utf-8")
            input_dict = {"file": base64_str, "model": "openai/whisper-small", "language": "english"}
            tool_result = await session.call_tool(
                "audio_to_text",
                input_dict,
            )
            result_content = tool_result.content
            # Check result
            if json.loads(result_content[0].text)["text"].startswith("who is"):
                print("Result correct.")
            else:
                print(f"Result wrong. Received was {result_content}")
                exit(1)


if __name__ == "__main__":
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    asyncio.run(validate_svc(ip_address, service_port))
