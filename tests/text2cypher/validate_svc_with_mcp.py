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

            # Test text2cypher conversion
            input_dict = {
                "input_text": "Get all persons who are older than 30 years",
                "conn_str": {"url": "bolt://localhost:7687", "username": "neo4j", "password": "neo4jtest"},
                "seeding": {
                    "use_case": "Movie dataset",
                    "data": [
                        {"type": "Person", "ids": ["Tom Hanks", "Emma Stone", "Ryan Gosling"], "ages": [68, 35, 43]}
                    ],
                },
            }

            tool_result = await session.call_tool(
                "text2cypher",
                input_dict,
            )
            result_content = tool_result.content

            # Check result - should contain a Cypher query
            if result_content:
                result_text = result_content[0].text
                print(f"Generated Cypher query: {result_text}")

                # Verify the result contains a valid Cypher query pattern
                # Common patterns for "persons older than 30"
                valid_patterns = ["MATCH", "WHERE", "age > 30", "Person", "RETURN"]

                # Check if at least some patterns are present in the query
                patterns_found = sum(1 for pattern in valid_patterns if pattern.lower() in result_text.lower())

                if patterns_found >= 3:  # At least 3 patterns should match
                    print("Result correct. Valid Cypher query generated.")
                else:
                    print(f"Result suspicious. Generated query may not be valid: {result_text}")
                    exit(1)
            else:
                print(f"Result wrong. No content returned. Received was {result_content}")
                exit(1)

            # Test with a more complex query
            complex_input = {"input_text": "Find all movies directed by people born after 1970"}

            complex_result = await session.call_tool(
                "text2cypher",
                complex_input,
            )
            complex_content = complex_result.content

            if complex_content:
                complex_text = complex_content[0].text
                print(f"Complex query result: {complex_text}")

                # Check for movie/director related patterns
                movie_patterns = ["Movie", "directed", "born", "1970", "MATCH", "WHERE"]
                movie_patterns_found = sum(1 for pattern in movie_patterns if pattern.lower() in complex_text.lower())

                if movie_patterns_found >= 3:
                    print("Complex query validation successful.")
                else:
                    print(f"Complex query validation failed. Patterns found: {movie_patterns_found}/6")
                    exit(1)
            else:
                print("Failed to generate complex query")
                exit(1)


if __name__ == "__main__":
    ip_address = sys.argv[1]
    service_port = sys.argv[2]
    asyncio.run(validate_svc(ip_address, service_port))
