# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/modelcontextprotocol/quickstart-resources/blob/main/mcp-client/client.py

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

# from anthropic import Anthropic
from ollama import Client

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = Client(
            host=os.environ["OLLAMA_ENDPOINT"] if "OLLAMA_ENDPOINT" in os.environ else "http://localhost:11434",
        )
        self.model = "qwen2.5-coder"
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.google_cse_id = os.environ.get("GOOGLE_CSE_ID")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server.

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        # Check https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/stdio.py#L34
        # https://github.com/chrishayuk/mcp-cli/blob/main/src/mcpcli/environment.py#L25 for env variables setup
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=get_default_environment()
            | {
                "http_proxy": os.environ.get("http_proxy"),
                "https_proxy": os.environ.get("https_proxy"),
                "no_proxy": os.environ.get("no_proxy"),
            },
        )

        # TODO (check this) start the server and establish the connection between client/server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools."""
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
<<<<<<< HEAD:comps/web_retrievers/src/mcp_google_search_client.py
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
=======
        available_tools = [
            {"name": tool.name, "description": tool.description, "input_schema": tool.inputSchema}
            for tool in response.tools
        ]
        # print(available_tools)
>>>>>>> 0e2891343fb69a40ea32e1a4e2bf52f58e653b6b:comps/mcp_google_search/src/google_search/client.py

        tool_names = [tool["name"] for tool in available_tools]

        messages.append(
            {
                "role": "system",
                "content": f"Please use only one tool name relevant to {query} in following: {tool_names}",
            }
        )

        # Initial Claude API call
        # response = self.anthropic.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     max_tokens=1000,
        #     messages=messages,
        #     tools=available_tools
        # )
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            tools=available_tools or [],
        )

        # Process response and handle tool calls
        final_text = []

        message = response.message

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool in message.tool_calls:
                # Execute tool call
                print("tool call started >>>>>")
                if tool.function.name in ["get-google-search-answer"]:
                    tool.function.arguments["google_api_key"] = self.google_api_key
                    tool.function.arguments["google_cse_id"] = self.google_cse_id
                result = await self.session.call_tool(tool.function.name, tool.function.arguments)
                print("<<<<< tool call ended.")
                # print(result)

                # directly append FIXME
                final_text.append(result.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
