#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

SAMPLE_TEXT = (
    "Alan Turing was born in London. "
    "Turing invented the Turing machine. "
    "Apple was founded by Steve Jobs in California. "
    "The Eiffel Tower is located in Paris."
)


def extract_text(content):
    if not content:
        return ""
    first = content[0]
    if hasattr(first, "text"):
        return first.text
    if isinstance(first, str):
        return first
    return str(first)


def has_triplet_keys(text):
    lowered = text.lower()
    if all(key in lowered for key in ["head", "tail", "type"]):
        return True
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False

    def walk(value):
        if isinstance(value, dict):
            keys = {k.lower() for k in value.keys()}
            if {"head", "tail", "type"}.issubset(keys):
                return True
            return any(walk(v) for v in value.values())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(payload)


async def validate_text2graph_mcp(ip_address, service_port):
    endpoint = f"http://{ip_address}:{service_port}"
    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tool_result = await session.call_tool("execute_agent", {"input_text": SAMPLE_TEXT})
            response_text = extract_text(tool_result.content)
            if not response_text:
                print("No response content from MCP tool.")
                return False

            if has_triplet_keys(response_text):
                print("MCP text2graph returned triplets.")
                return True

            print(f"Unexpected response content: {response_text}")
            return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    success = asyncio.run(validate_text2graph_mcp(ip_address, service_port))
    sys.exit(0 if success else 1)
