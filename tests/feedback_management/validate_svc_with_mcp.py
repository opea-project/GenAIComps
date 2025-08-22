#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def validate_svc(ip_address, service_port):

    endpoint = f"http://{ip_address}:{service_port}"

    async with sse_client(endpoint + "/sse") as streams:
        async with ClientSession(*streams) as session:
            result = await session.initialize()

            # Test creating feedback
            input_dict = {
                "feedback": {
                    "chat_id": "test_mcp_chat_001",
                    "chat_data": {
                        "user": "test_mcp_user",
                        "messages": [
                            {"role": "user", "content": "What is OPEA?"},
                            {
                                "role": "assistant",
                                "content": "OPEA (Open Platform for Enterprise AI) is an open-source framework for building enterprise-grade Generative AI applications.",
                            },
                        ],
                    },
                    "feedback_data": {
                        "comment": "Great response! Very informative.",
                        "rating": 5,
                        "is_thumbs_up": True,
                    },
                }
            }
            tool_result = await session.call_tool(
                "create_feedback_data",
                input_dict,
            )
            result_content = tool_result.content

            # Check result - feedback_id should be returned as a string
            feedback_id = result_content[0].text
            if feedback_id and len(feedback_id) == 24:  # MongoDB ObjectId is 24 characters
                print(f"Create feedback successful. Feedback ID: {feedback_id}")

                # Test retrieving feedback
                retrieve_input = {"feedback": {"user": "test_mcp_user", "feedback_id": feedback_id}}
                retrieve_result = await session.call_tool(
                    "get_feedback",
                    retrieve_input,
                )
                retrieve_content = retrieve_result.content
                try:
                    # Parse the JSON response
                    feedback_data = json.loads(retrieve_content[0].text)
                except json.JSONDecodeError:
                    # If it's not JSON, it might be returned as a Python dict string
                    import ast

                    feedback_data = ast.literal_eval(retrieve_content[0].text)

                # Verify the retrieved feedback matches what we created
                if (
                    feedback_data.get("chat_id") == "test_mcp_chat_001"
                    and feedback_data.get("feedback_data", {}).get("rating") == 5
                    and feedback_data.get("feedback_data", {}).get("is_thumbs_up") is True
                ):
                    print("Result correct. Feedback retrieved successfully.")
                else:
                    print(f"Result wrong. Retrieved feedback doesn't match. Received was {feedback_data}")
                    exit(1)
            else:
                print(f"Result wrong. Invalid feedback_id. Received was {result_content}")
                exit(1)


if __name__ == "__main__":
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    asyncio.run(validate_svc(ip_address, service_port))
