#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

SAMPLE_INPUT = {
    "prompt": "A small red robot holding a yellow balloon",
    "num_images_per_prompt": 1,
}


def extract_text(content):
    if not content:
        return ""
    first = content[0]
    if hasattr(first, "text"):
        return first.text
    if isinstance(first, str):
        return first
    return str(first)


def has_images(payload):
    if isinstance(payload, dict):
        images = payload.get("images")
        return isinstance(images, list) and len(images) > 0
    return False


async def validate_text2image_mcp(ip_address, service_port):
    endpoint = f"http://{ip_address}:{service_port}"
    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tool_result = await session.call_tool("text2image", {"input": SAMPLE_INPUT})
            response_text = extract_text(tool_result.content)
            if not response_text:
                print("No response content from MCP tool.")
                return False

            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError:
                try:
                    import ast

                    payload = ast.literal_eval(response_text)
                except Exception:
                    payload = None

            if payload and has_images(payload):
                print("MCP text2image returned images.")
                return True

            if "images" in response_text.lower():
                print("MCP text2image returned images (string match).")
                return True

            print(f"Unexpected response content: {response_text}")
            return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    success = asyncio.run(validate_text2image_mcp(ip_address, service_port))
    sys.exit(0 if success else 1)
