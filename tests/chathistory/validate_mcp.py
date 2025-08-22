#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def validate_chathistory_mcp(ip_address, service_port):
    """Validate ChatHistory service MCP functionality."""
    endpoint = f"http://{ip_address}:{service_port}"

    # Ensure trailing slash for SSE endpoint to avoid redirect
    sse_endpoint = endpoint + "/sse/"
    print(f"Connecting to SSE endpoint: {sse_endpoint}")

    async with sse_client(sse_endpoint) as streams:
        async with ClientSession(*streams) as session:
            result = await session.initialize()

            # Test create operation
            chat_data = {
                "data": {
                    "messages": [{"role": "user", "content": "Hello, this is a test message"}],
                    "user": "test_user",
                },
                "first_query": "Hello, this is a test message",
            }

            # Create chat conversation
            tool_result = await session.call_tool(
                "create_documents",
                {"document": chat_data},
            )
            create_result = tool_result.content

            # The response is just the ID string, not JSON
            conversation_id = create_result[0].text

            if not conversation_id:
                print(f"Create operation failed. Received was {create_result}")
                exit(1)

            # Test get operation
            get_data = {"user": "test_user", "id": conversation_id}

            tool_result = await session.call_tool(
                "get_documents",
                {"document": get_data},
            )
            get_result = tool_result.content
            # Try to parse the response as JSON
            try:
                retrieved_doc = json.loads(get_result[0].text)
            except json.JSONDecodeError:
                # If not JSON, assume it's an error or empty
                retrieved_doc = None

            if not retrieved_doc or retrieved_doc.get("user") != "test_user":
                print(f"Get operation failed. Received was {get_result}")
                exit(1)

            # Test delete operation
            delete_data = {"user": "test_user", "id": conversation_id}

            tool_result = await session.call_tool(
                "delete_documents",
                {"document": delete_data},
            )
            delete_result = tool_result.content

            if not delete_result:
                print(f"Delete operation failed. Received was {delete_result}")
                exit(1)

            print("Result correct.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        exit(1)
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    asyncio.run(validate_chathistory_mcp(ip_address, service_port))
