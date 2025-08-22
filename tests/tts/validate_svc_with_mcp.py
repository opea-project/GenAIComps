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
            input_dict = {"request": {"input": "Hi there, welcome to OPEA."}}
            tool_result = await session.call_tool(
                "text_to_speech",
                input_dict,
            )
            result_content = tool_result.content
            # Check result
            audio_str = json.loads(result_content[0].text).get("audio_str", "")
            if audio_str.startswith("Ukl"):  # "Ukl" indicates likely WAV header
                audio_data = base64.b64decode(audio_str)
                with open("output.wav", "wb") as f:
                    f.write(audio_data)
                with open("output.wav", "rb") as f:
                    header = f.read(4)
                if header == b"RIFF":
                    print("Result correct.")
                else:
                    print(f"Invalid WAV file: starts with {header}")
            else:
                print(f"Result wrong. Received was {result_content}")
                exit(1)


if __name__ == "__main__":
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    asyncio.run(validate_svc(ip_address, service_port))
