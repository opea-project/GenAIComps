# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from contextlib import AsyncExitStack
from typing import List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from comps import CustomLogger
from comps.cores.mcp.tool import OpeaMCPClientTool

logger = CustomLogger("comps-mcp-client")
log_flag = os.getenv("LOGFLAG", False)


class OpeaMCPClient(BaseModel):
    """A client for interacting with MCP servers, managing tools, and handling server communication."""

    description: str = "MCP client for server interaction and tool management"
    session: Optional[ClientSession] = None
    exit_stack: AsyncExitStack = AsyncExitStack()

    tools: List[OpeaMCPClientTool] = Field(default_factory=list)
    tool_registry: dict[str, OpeaMCPClientTool] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    async def connect_via_sse(self, server_url: str, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        """Establish a connection to an MCP server using SSE (Server-Sent Events) transport.

        Args:
            server_url: The URL of the SSE server to connect to.
            api_key: Optional API key for authentication.
            timeout: Connection timeout in seconds. Default is 30 seconds.

        Raises:
            ValueError: If the server URL is not provided.
            asyncio.TimeoutError: If the connection times out.
            Exception: For other connection errors.
        """
        if not server_url:
            raise ValueError("Server URL is required.")
        if self.session:
            await self.disconnect()

        try:

            async def connect_with_timeout():
                streams_context = sse_client(
                    url=server_url,
                    headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
                    timeout=timeout,
                )
                streams = await self.exit_stack.enter_async_context(streams_context)
                self.session = await self.exit_stack.enter_async_context(ClientSession(*streams))
                await self._initialize_tools()

            await asyncio.wait_for(connect_with_timeout(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Connection to {server_url} timed out after {timeout} seconds")
            await self.disconnect()
            raise
        except Exception as e:
            logger.error(f"Error connecting to {server_url}: {str(e)}")
            await self.disconnect()
            raise

    async def connect_via_stdio(self, command: str, args: List[str]) -> None:
        """Establish a connection to an MCP server using stdio (standard input/output) transport.

        Args:
            command: The command to start the server.
            args: A list of arguments for the command.

        Raises:
            ValueError: If the command is not provided.
            Exception: For other connection errors.
        """
        if not command:
            raise ValueError("Server command is required.")
        if self.session:
            await self.disconnect()

        try:
            server_params = StdioServerParameters(command=command, args=args)
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))

            await self._initialize_tools()
        except Exception as e:
            logger.error(f"Error connecting to {command}: {str(e)}")
            await self.disconnect()
            raise

    async def _initialize_tools(self) -> None:
        """Initialize the client session and populate the tool registry with available tools.

        Raises:
            RuntimeError: If the session is not initialized.
        """
        if not self.session:
            raise RuntimeError("Session not initialized.")

        await self.session.initialize()
        response = await self.session.list_tools()

        # Clear existing tools
        self.tools = []
        self.tool_registry = {}

        # Populate tools and registry
        for tool in response.tools:
            client_tool = OpeaMCPClientTool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.inputSchema,
                session=self.session,
            )
            self.tool_registry[tool.name] = client_tool
            self.tools.append(client_tool)

        logger.info(f"Connected to server with tools: {[tool.name for tool in response.tools]}")

    async def invoke_tool(self, tool_name: str, parameters: dict):
        """Invoke a tool on the MCP server.

        Args:
            tool_name: The name of the tool to invoke.
            parameters: The parameters to pass to the tool.

        Returns:
            The result of the tool invocation.

        Raises:
            ValueError: If the tool is not found in the registry.
            RuntimeError: If the client session is not available.
        """
        if tool_name not in self.tool_registry:
            raise ValueError(f"Tool '{tool_name}' not found in the registry.")
        if not self.session:
            raise RuntimeError("Client session is not available.")

        return await self.session.call_tool(name=tool_name, arguments=parameters)

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        if self.session:
            try:
                if hasattr(self.session, "close"):
                    await self.session.close()
                await self.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error during disconnect: {str(e)}")
            finally:
                self.session = None
                self.tools = []
                self.tool_registry = {}
                logger.info("Disconnected from MCP server")
