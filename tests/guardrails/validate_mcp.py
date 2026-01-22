#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MCP validation script for guardrails service."""

import asyncio
import os
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def _get_tools_from_init(result):
    tools = []
    if hasattr(result, "capabilities") and hasattr(result.capabilities, "tools"):
        tools_data = result.capabilities.tools
        if isinstance(tools_data, dict):
            tools = list(tools_data.keys())
        elif isinstance(tools_data, list):
            if tools_data and hasattr(tools_data[0], "name"):
                tools = [tool.name for tool in tools_data]
            else:
                tools = tools_data
    return tools


async def validate_guardrails_mcp(ip_address, service_port):
    endpoint = f"http://{ip_address}:{service_port}"

    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            result = await session.initialize()

            tools = _get_tools_from_init(result)
            if not tools:
                tool_list = await session.list_tools()
                if hasattr(tool_list, "tools"):
                    tools_data = tool_list.tools
                    if isinstance(tools_data, list):
                        if tools_data and hasattr(tools_data[0], "name"):
                            tools = [tool.name for tool in tools_data]
                        else:
                            tools = tools_data

            expected_tools = ["safety_guard"]
            missing_tools = [t for t in expected_tools if t not in tools]
            if missing_tools:
                print(f"Missing expected tools: {missing_tools}")
                return False

            if os.getenv("GUARDRAILS_MCP_SMOKE_CALL", "").strip().lower() in {"1", "true", "yes"}:
                tool_result = await session.call_tool(
                    "safety_guard",
                    {"input": {"text": "Hello from MCP"}},
                )
                if not tool_result.content:
                    print("Guardrails tool call failed: no content returned")
                    return False

            print("Guardrails MCP validation passed")
            return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]

    success = asyncio.run(validate_guardrails_mcp(ip_address, service_port))
    sys.exit(0 if success else 1)
