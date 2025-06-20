# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Union

from comps import CustomLogger
from comps.cores.mcp.client import OpeaMCPClient
from comps.cores.mcp.config import OpeaMCPConfig, OpeaMCPSSEServerConfig, OpeaMCPStdioServerConfig

logger = CustomLogger("comps-mcp-manager")
logflag = os.getenv("LOGFLAG", False)


class OpeaMCPToolsManager:
    """A unified interface for handling MCP clients with different server configurations."""

    def __init__(self, config: OpeaMCPConfig):
        """Initialize the MCPToolsManager with the provided configuration.

        Args:
            config: The OPEA MCP configuration containing server details.
        """
        self.config = config
        self.tools_registry: List[dict] = []
        self.clients: List[OpeaMCPClient] = []

    @classmethod
    async def create(cls, config: OpeaMCPConfig) -> "OpeaMCPToolsManager":
        """Asynchronous factory method to create an instance of OpeaMCPToolsManager.

        Args:
            config: The OPEA MCP configuration containing server details.

        Returns:
            An instance of OpeaMCPToolsManager with initialized clients.
        """
        instance = cls(config)
        instance.clients = await instance._initialize_clients(config.sse_servers + config.stdio_servers)
        await instance._register_tools()
        return instance

    async def _initialize_clients(
        self, server_configs: List[Union[OpeaMCPSSEServerConfig, OpeaMCPStdioServerConfig]]
    ) -> List[OpeaMCPClient]:
        """Initialize MCP clients based on the provided server configurations.

        Args:
            server_configs: A list of server configurations (SSE or Stdio).

        Returns:
            A list of initialized MCP clients.
        """
        initialized_clients = []
        for server_config in server_configs:
            client = OpeaMCPClient()
            try:
                if isinstance(server_config, OpeaMCPSSEServerConfig):
                    logger.info(f"Initializing MCP client for SSE server: {server_config.url}")
                    await client.connect_via_sse(server_config.url, server_config.api_key)
                elif isinstance(server_config, OpeaMCPStdioServerConfig):
                    logger.info(f"Initializing MCP client for Stdio server: {server_config.command}")
                    await client.connect_via_stdio(server_config.command, server_config.args)
                else:
                    logger.error(f"Unsupported server configuration type: {server_config}")
                    continue

                initialized_clients.append(client)
                logger.info(f"Successfully connected to MCP server: {server_config}")
            except Exception as e:
                logger.error(f"Failed to connect to server {server_config}: {str(e)}")
                try:
                    await client.disconnect()
                except Exception as disconnect_error:
                    logger.error(f"Error during disconnect after failed connection: {str(disconnect_error)}")

        return initialized_clients

    async def _register_tools(self):
        """Dynamically register tools as methods of the manager for natural invocation."""
        tools = self._extract_tools_from_clients(self.clients)
        for tool in tools:
            tool_name = tool.get("function", {}).get("name")
            if not tool_name:
                logger.error(f"Tool metadata missing 'name': {tool}")
                continue

            async def tool_method(self, **kwargs):
                return await self.execute_tool(tool_name, kwargs)

            # Dynamically add the tool method to the manager
            setattr(self, tool_name, tool_method.__get__(self))

    def _extract_tools_from_clients(self, clients: List[OpeaMCPClient]) -> List[dict]:
        """Extracts tools from a list of OpeaMCPClient instances and converts them to a standardized format.

        Args:
            clients: List of OpeaMCPClient instances.

        Returns:
            A list of tool dictionaries ready to be used by OPEA Agents.
        """
        if not clients:
            logger.warning("No MCP clients provided, returning an empty tool list.")
            return []

        try:
            for client in clients:
                for tool in client.tools:
                    tool_metadata = tool.to_param()
                    self.tools_registry.append(tool_metadata)
        except Exception as e:
            logger.error(f"Error while extracting tools from clients: {e}")
            return []
        return self.tools_registry

    async def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute a tool on an MCP server and return the result.

        Args:
            tool_name: The name of the tool to execute.
            parameters: The parameters to pass to the tool.

        Returns:
            The result of the tool execution as a JSON string.
        """
        if not self.clients:
            raise ValueError("No MCP clients are currently connected.")

        logger.debug(f"Attempting to execute tool: {tool_name}")

        # Find the client that provides the specified tool
        target_client = None
        for client in self.clients:
            logger.debug(f"Checking tools for client: {client}")
            if tool_name in [tool.name for tool in client.tools]:
                target_client = client
                break

        if target_client is None:
            raise ValueError(f"No MCP client found that provides the tool: {tool_name}")

        logger.debug(f"Found matching client for tool {tool_name}: {target_client}")

        # Execute the tool
        response = await target_client.invoke_tool(tool_name, parameters)
        logger.debug(f"Received response from tool {tool_name}: {response}")

        return json.dumps(response.model_dump(mode="json"))

    async def __aenter__(self):
        """Support for async context management."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Disconnect all clients on exit."""
        for client in self.clients:
            try:
                await client.disconnect()
            except Exception as disconnect_error:
                logger.error(f"Error while disconnecting MCP client: {str(disconnect_error)}")
