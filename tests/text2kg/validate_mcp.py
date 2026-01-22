#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

SAMPLE_TEXT = "Who is Paul Graham?"


def extract_text(content):
    if not content:
        return ""
    first = content[0]
    if hasattr(first, "text"):
        return first.text
    if isinstance(first, str):
        return first
    return str(first)


def has_result(payload):
    if isinstance(payload, dict):
        if "result" in payload or "output" in payload:
            return True
    return False


async def validate_text2kg_mcp(ip_address, service_port):
    endpoint = f"http://{ip_address}:{service_port}"
    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tool_result = await session.call_tool("execute_agent", {"input_text": SAMPLE_TEXT})
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

            if payload and has_result(payload):
                print("MCP text2kg returned a result.")
                return True

            if response_text.strip():
                print("MCP text2kg returned text.")
                return True

            print(f"Unexpected response content: {response_text}")
            return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    success = asyncio.run(validate_text2kg_mcp(ip_address, service_port))
    sys.exit(0 if success else 1)
