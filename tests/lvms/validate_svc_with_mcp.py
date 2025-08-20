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
            input_dict = {
                "request": {
                    "image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC",
                    "prompt": "What is this?",
                }
            }
            tool_result = await session.call_tool("lvm", input_dict)
            result_content = tool_result.content
            res = json.loads(result_content[0].text).get("text", None)
            if res and "yellow" in res:
                print("Result correct.")
            else:
                print(f"Result wrong. Received was {result_content}")


if __name__ == "__main__":
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    asyncio.run(validate_svc(ip_address, service_port))
