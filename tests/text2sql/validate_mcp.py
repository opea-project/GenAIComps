#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def extract_text(content):
    if not content:
        return ""
    first = content[0]
    if hasattr(first, "text"):
        return first.text
    if isinstance(first, str):
        return first
    return str(first)


def parse_payload(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import ast

            return ast.literal_eval(text)
        except Exception:
            return None


async def validate_text2sql_mcp(ip_address, service_port, db_config):
    endpoint = f"http://{ip_address}:{service_port}"
    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tool_result = await session.call_tool(
                "execute_agent",
                {
                    "input": {
                        "input_text": "Find the total number of Albums.",
                        "conn_str": db_config,
                    }
                },
            )
            response_text = extract_text(tool_result.content)
            if not response_text:
                print("No response content from MCP tool.")
                return False

            payload = parse_payload(response_text)
            if isinstance(payload, dict) and ("result" in payload or "output" in payload):
                print("MCP text2sql returned a result.")
                return True

            if "output" in response_text.lower():
                print("MCP text2sql returned output (string match).")
                return True

            print(f"Unexpected response content: {response_text}")
            return False


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 validate_mcp.py <ip> <port> <user> <password> <host> <db>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    db_config = {
        "user": sys.argv[3],
        "password": sys.argv[4],
        "host": sys.argv[5],
        "port": 5442,
        "database": sys.argv[6],
    }

    success = asyncio.run(validate_text2sql_mcp(ip_address, service_port, db_config))
    sys.exit(0 if success else 1)
