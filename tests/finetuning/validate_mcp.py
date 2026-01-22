#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MCP validation script for finetuning service."""

import asyncio
import base64
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def _extract_text(content):
    if not content:
        return None
    first = content[0]
    if hasattr(first, "text"):
        return first.text
    if isinstance(first, str):
        return first
    return None


def _parse_payload(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            return ast.literal_eval(text)
        except Exception:
            return None


async def validate_finetuning_mcp(ip_address, service_port):
    endpoint = f"http://{ip_address}:{service_port}"

    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            result = await session.initialize()

            tools = []
            try:
                if hasattr(result, "capabilities") and hasattr(result.capabilities, "tools"):
                    tools_data = result.capabilities.tools
                    if isinstance(tools_data, dict):
                        tools = list(tools_data.keys())
                    elif isinstance(tools_data, list):
                        if tools_data and hasattr(tools_data[0], "name"):
                            tools = [tool.name for tool in tools_data]
                        else:
                            tools = tools_data

                if not tools:
                    tool_list = await session.list_tools()
                    if hasattr(tool_list, "tools"):
                        tools_data = tool_list.tools
                        if isinstance(tools_data, list):
                            if tools_data and hasattr(tools_data[0], "name"):
                                tools = [tool.name for tool in tools_data]
                            else:
                                tools = tools_data
            except Exception as exc:
                print(f"Error getting tools: {exc}")
                print("Will proceed without tool validation")

            expected_tools = [
                "create_finetuning_jobs",
                "list_finetuning_jobs",
                "retrieve_finetuning_job",
                "cancel_finetuning_job",
                "upload_training_files_mcp",
                "list_checkpoints",
            ]

            missing_tools = [t for t in expected_tools if t not in tools]
            if missing_tools:
                print(f"Missing expected tools: {missing_tools}")
                return False

            # Test base64 upload tool
            filename = "mcp_test_finetune.jsonl"
            payload = '{"instruction":"hello","input":"","output":"world"}\n'
            encoded = base64.b64encode(payload.encode("utf-8")).decode("utf-8")

            upload_result = await session.call_tool(
                "upload_training_files_mcp",
                {"request": {"filename": filename, "content_base64": encoded, "purpose": "fine-tune"}},
            )
            upload_text = _extract_text(upload_result.content)
            upload_data = _parse_payload(upload_text)

            if upload_data is None:
                print(f"Upload failed: unexpected response {upload_result.content}")
                return False

            if upload_data.get("filename") != filename:
                print(f"Upload failed: wrong filename. Got: {upload_data}")
                return False

            # Test list jobs tool
            list_result = await session.call_tool("list_finetuning_jobs", {})
            list_text = _extract_text(list_result.content)
            list_data = _parse_payload(list_text)

            if list_data is None or "data" not in list_data:
                print(f"List jobs failed: unexpected response {list_result.content}")
                return False

            print("Finetuning MCP validation passed")
            return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]

    success = asyncio.run(validate_finetuning_mcp(ip_address, service_port))
    sys.exit(0 if success else 1)
