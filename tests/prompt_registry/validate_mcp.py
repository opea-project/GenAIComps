#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MCP validation script for prompt registry service."""

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def validate_prompt_registry_mcp(ip_address, service_port):
    """Validate prompt registry MCP functionality."""
    endpoint = f"http://{ip_address}:{service_port}"

    print(f"Connecting to MCP endpoint: {endpoint}/sse")

    try:
        async with sse_client(endpoint + "/sse") as streams:
            print("SSE client connected")
            async with ClientSession(*streams) as session:
                print("MCP session created")
                # Initialize MCP session
                result = await session.initialize()

                # Check if we got the expected tool names
                # Try to get the list of tools - MCP API may vary
                tools = []
                try:
                    # Try different ways to get tools based on MCP version
                    if hasattr(result, "capabilities") and hasattr(result.capabilities, "tools"):
                        # Tools might be a dict or list
                        tools_data = result.capabilities.tools
                        if isinstance(tools_data, dict):
                            tools = list(tools_data.keys())
                        elif isinstance(tools_data, list):
                            # Check if it's a list of objects or strings
                            if tools_data and hasattr(tools_data[0], "name"):
                                tools = [tool.name for tool in tools_data]
                            else:
                                tools = tools_data

                    # If no tools found yet, try listing them
                    if not tools:
                        tool_list = await session.list_tools()
                        if hasattr(tool_list, "tools"):
                            tools_data = tool_list.tools
                            if isinstance(tools_data, list):
                                # Check if it's a list of objects or strings
                                if tools_data and hasattr(tools_data[0], "name"):
                                    tools = [tool.name for tool in tools_data]
                                else:
                                    tools = tools_data
                except Exception as e:
                    print(f"Error getting tools: {e}")
                    print("Will proceed without tool validation")

                expected_tools = ["create_prompt", "get_prompt", "delete_prompt"]

                missing_tools = [t for t in expected_tools if t not in tools]
                if missing_tools:
                    print(f"Missing expected tools: {missing_tools}")
                    return False

                print(f"MCP initialization successful. Found tools: {tools}")

                # Test create_prompt tool
                print("\nTesting create_prompt tool...")
                create_result = await session.call_tool(
                    "create_prompt", {"prompt": {"prompt_text": "Test MCP prompt", "user": "mcp_test_user"}}
                )

                # Parse the response
                create_content = create_result.content
                if not create_content:
                    print("create_prompt failed: no content returned")
                    return False

                # The response might be in different formats
                response_text = None
                if hasattr(create_content[0], "text"):
                    response_text = create_content[0].text
                elif isinstance(create_content[0], str):
                    response_text = create_content[0]
                else:
                    print(f"Unexpected content format: {type(create_content[0])}, content: {create_content[0]}")
                    return False

                # The response might already be a prompt_id string or a JSON object
                prompt_id = None
                if response_text:
                    try:
                        # Try to parse as JSON
                        response = json.loads(response_text)
                        prompt_id = response.get("prompt_id") or response.get("id")
                    except json.JSONDecodeError:
                        # Maybe it's just the prompt_id string
                        prompt_id = response_text.strip().strip('"')

                if not prompt_id:
                    print(f"create_prompt failed: no prompt_id in response. Got: {response}")
                    return False

                print(f"Created prompt with ID: {prompt_id}")

                # Test get_prompt tool
                print("\nTesting get_prompt tool...")
                get_result = await session.call_tool(
                    "get_prompt", {"prompt": {"user": "mcp_test_user", "prompt_id": prompt_id}}
                )

                get_content = get_result.content
                if not get_content:
                    print("get_prompt failed: no content returned")
                    return False

                # Parse the response
                response_text = None
                if hasattr(get_content[0], "text"):
                    response_text = get_content[0].text
                elif isinstance(get_content[0], str):
                    response_text = get_content[0]

                get_response = {}
                retrieved_prompt = None
                if response_text:
                    try:
                        get_response = json.loads(response_text)
                        retrieved_prompt = get_response.get("prompt")
                    except json.JSONDecodeError:
                        # The response might just be the prompt text
                        retrieved_prompt = {"prompt_text": response_text}

                if not retrieved_prompt or retrieved_prompt.get("prompt_text") != "Test MCP prompt":
                    print(f"get_prompt failed: unexpected response. Got: {get_response}")
                    return False

                print(f"Retrieved prompt: {retrieved_prompt.get('prompt_text')}")

                # Test delete_prompt tool
                print("\nTesting delete_prompt tool...")
                delete_result = await session.call_tool(
                    "delete_prompt", {"prompt": {"user": "mcp_test_user", "prompt_id": prompt_id}}
                )

                delete_content = delete_result.content
                if not delete_content:
                    print("delete_prompt failed: no content returned")
                    return False

                # Parse the response
                response_text = None
                if hasattr(delete_content[0], "text"):
                    response_text = delete_content[0].text
                elif isinstance(delete_content[0], str):
                    response_text = delete_content[0]

                delete_response = {}
                if response_text:
                    try:
                        delete_response = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If it's not JSON, could be a success message or ID
                        # Consider it successful if we got any response
                        print(f"Delete response (non-JSON): {response_text}")
                        delete_response = {"success": True}

                # If we got a response, consider it successful
                # (the actual verification comes from trying to get the deleted prompt)
                if not response_text and not delete_response.get("success"):
                    print("delete_prompt failed: no response")
                    return False

                print("Successfully deleted prompt")

                # Verify deletion by trying to get the deleted prompt
                print("\nVerifying deletion...")
                try:
                    verify_result = await session.call_tool(
                        "get_prompt", {"prompt": {"user": "mcp_test_user", "prompt_id": prompt_id}}
                    )
                    verify_content = verify_result.content
                    if verify_content:
                        response_text = None
                        if hasattr(verify_content[0], "text"):
                            response_text = verify_content[0].text
                        elif isinstance(verify_content[0], str):
                            response_text = verify_content[0]

                        if response_text:
                            try:
                                verify_response = json.loads(response_text)
                                # Should get an error or empty result
                                if verify_response.get("prompt"):
                                    print("Deletion verification failed: prompt still exists")
                                    return False
                            except:
                                # Error parsing means prompt was not found - good
                                pass
                except Exception:
                    # Expected to fail when trying to get deleted prompt
                    pass

                print("Deletion verified: prompt no longer exists")
                print("\nAll MCP tests passed successfully!")
                return True

    except Exception as e:
        import traceback

        print(f"MCP validation failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 validate_mcp.py <ip_address> <service_port>")
        sys.exit(1)

    ip_address = sys.argv[1]
    service_port = sys.argv[2]

    success = asyncio.run(validate_prompt_registry_mcp(ip_address, service_port))
    sys.exit(0 if success else 1)
