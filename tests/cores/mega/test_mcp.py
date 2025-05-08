# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, MagicMock
import asyncio
from comps.agent.src.tools.mcp.client import OpeaMCPClient
from comps.agent.src.tools.mcp.config import OpeaMCPConfig, OpeaMCPSSEServerConfig, OpeaMCPStdioServerConfig
from comps.agent.src.tools.mcp.manager import OpeaMCPToolsManager


class TestOpeaMCPToolsManager(unittest.TestCase):
    def setUp(self):
        # Create mock configurations for the servers
        self.mock_sse_config = OpeaMCPSSEServerConfig(url="http://sse-server.com", api_key="dummy_api_key")
        self.mock_stdio_config = OpeaMCPStdioServerConfig(name="test", command="python", args=["tool.py"])

        # Mock the OpeaMCPConfig to return these configurations
        self.mock_config = MagicMock(spec=OpeaMCPConfig)
        self.mock_config.sse_servers = [self.mock_sse_config]
        self.mock_config.stdio_servers = [self.mock_stdio_config]

        # Create the OpeaMCPToolsManager instance using the mock configuration
        self.manager = OpeaMCPToolsManager(self.mock_config)

    @patch.object(OpeaMCPToolsManager, '_initialize_clients', return_value=[])
    @patch.object(OpeaMCPToolsManager, '_register_tools')
    def test_create_success(self, mock_register_tools, mock_initialize_clients):
        """Test that OpeaMCPToolsManager can be successfully created"""
        asyncio.run(self.manager.create(self.mock_config))

        mock_initialize_clients.assert_called_once()
        mock_register_tools.assert_called_once()

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    @patch('comps.agent.src.tools.mcp.manager.logger')
    async def test_initialize_clients_success(self, mock_logger, MockOpeaMCPClient):
        """Test the _initialize_clients method with valid configurations"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client

        # Mock successful connections
        mock_client.connect_via_sse = MagicMock()
        mock_client.connect_via_stdio = MagicMock()

        initialized_clients = await self.manager._initialize_clients([self.mock_sse_config, self.mock_stdio_config])

        # Assert that the client was initialized successfully
        self.assertEqual(len(initialized_clients), 2)
        mock_client.connect_via_sse.assert_called_once_with(self.mock_sse_config.url, self.mock_sse_config.api_key)
        mock_client.connect_via_stdio.assert_called_once_with(self.mock_stdio_config.command, self.mock_stdio_config.args)

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    @patch('comps.agent.src.tools.mcp.manager.logger')
    async def test_initialize_clients_failure(self, mock_logger, MockOpeaMCPClient):
        """Test the _initialize_clients method when a client connection fails"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client

        # Simulate an error during connection
        mock_client.connect_via_sse.side_effect = Exception("Connection failed")

        initialized_clients = await self.manager._initialize_clients([self.mock_sse_config])

        # Assert that no clients were initialized due to the failure
        self.assertEqual(len(initialized_clients), 0)

    @patch.object(OpeaMCPToolsManager, '_extract_tools_from_clients', return_value=[{'name': 'mock_tool', 'params': {}}])
    @patch('comps.agent.src.tools.mcp.manager.logger')
    async def test_register_tools(self, mock_logger, mock_extract_tools):
        """Test tool registration"""
        await self.manager._register_tools()

        # Assert that _extract_tools_from_clients was called once
        mock_extract_tools.assert_called_once()
        # Check that a method has been dynamically added for the mock tool
        self.assertTrue(hasattr(self.manager, 'mock_tool'))

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    @patch('comps.agent.src.tools.mcp.manager.logger')
    async def test_execute_tool_success(self, mock_logger, MockOpeaMCPClient):
        """Test executing a tool on an MCP client"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.invoke_tool = MagicMock(return_value={"result": "success"})

        # Add a mock tool to the client
        mock_client.tools = [MagicMock(name="mock_tool", to_param=MagicMock(return_value={"name": "mock_tool"}))]

        # Call the method to execute the tool
        result = await self.manager.execute_tool("mock_tool", {"param1": "value"})

        self.assertEqual(result, '{"result": "success"}')
        mock_client.invoke_tool.assert_called_once_with("mock_tool", {"param1": "value"})

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_execute_tool_no_clients(self, MockOpeaMCPClient):
        """Test executing a tool when no MCP clients are connected"""
        with self.assertRaises(ValueError) as context:
            await self.manager.execute_tool("mock_tool", {"param1": "value"})

        self.assertEqual(str(context.exception), 'No MCP clients are currently connected.')

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_execute_tool_tool_not_found(self, MockOpeaMCPClient):
        """Test executing a tool when the tool is not found in any client"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = []

        with self.assertRaises(ValueError) as context:
            await self.manager.execute_tool("mock_tool", {"param1": "value"})

        self.assertEqual(str(context.exception), 'No MCP client found that provides the tool: mock_tool')

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_execute_tool_with_sse(self, MockOpeaMCPClient):
        """Test tool execution when using an SSE server"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.invoke_tool = MagicMock(return_value={"result": "sse_tool_success"})

        # Add mock tool for SSE
        mock_client.tools = [MagicMock(name="sse_tool", to_param=MagicMock(return_value={"name": "sse_tool"}))]

        result = await self.manager.execute_tool("sse_tool", {"param1": "value"})

        self.assertEqual(result, '{"result": "sse_tool_success"}')
        mock_client.invoke_tool.assert_called_once_with("sse_tool", {"param1": "value"})

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_context_manager_disconnects(self, MockOpeaMCPClient):
        """Test that the context manager disconnects all clients"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        self.manager.clients = [mock_client]

        async with self.manager:
            pass  # Test that __aenter__ works

        mock_client.disconnect.assert_called_once()

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_initialize_clients_with_failure(self, MockOpeaMCPClient):
        """Test that clients are handled correctly when initialization fails"""
        MockOpeaMCPClient.return_value.connect_via_sse = MagicMock(side_effect=Exception("Connection failed"))

        # This should result in an empty client list after failure
        clients = await self.manager._initialize_clients([self.mock_sse_config])
        self.assertEqual(clients, [])

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_initialize_clients_partial_failure(self, MockOpeaMCPClient):
        """Test that _initialize_clients handles partial failures correctly"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client

        # Simulate one successful connection and one failure
        mock_client.connect_via_sse.side_effect = [None, Exception("Connection failed")]

        clients = await self.manager._initialize_clients([self.mock_sse_config, self.mock_sse_config])

        # Assert that only one client was initialized successfully
        self.assertEqual(len(clients), 1)
        mock_client.connect_via_sse.assert_called()

    @patch.object(OpeaMCPToolsManager, '_extract_tools_from_clients', return_value=[])
    async def test_register_tools_no_tools(self, mock_extract_tools):
        """Test _register_tools when no tools are available"""
        await self.manager._register_tools()

        # Assert that no tools were registered
        mock_extract_tools.assert_called_once()
        self.assertEqual(len(self.manager.tools_registry), 0)

    @patch.object(OpeaMCPToolsManager, '_extract_tools_from_clients', return_value=[{'name': 'duplicate_tool'}, {'name': 'duplicate_tool'}])
    async def test_register_tools_duplicate_tools(self, mock_extract_tools):
        """Test _register_tools with duplicate tool names"""
        await self.manager._register_tools()

        # Assert that duplicate tools are handled correctly
        self.assertEqual(len(self.manager.tools_registry), 2)
        self.assertTrue(hasattr(self.manager, 'duplicate_tool'))

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_context_manager_with_exception(self, MockOpeaMCPClient):
        """Test that the context manager handles exceptions correctly"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        self.manager.clients = [mock_client]

        with self.assertRaises(Exception):
            async with self.manager:
                raise Exception("Test exception")

        # Assert that disconnect was called even after an exception
        mock_client.disconnect.assert_called_once()

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_execute_tool_invalid_parameters(self, MockOpeaMCPClient):
        """Test execute_tool with invalid parameters"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = [MagicMock(name="mock_tool")]

        with self.assertRaises(TypeError):
            await self.manager.execute_tool("mock_tool", None)

    @patch('comps.agent.src.tools.mcp.manager.OpeaMCPClient')
    async def test_execute_tool_empty_tool_name(self, MockOpeaMCPClient):
        """Test execute_tool with an empty tool name"""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = []

        with self.assertRaises(ValueError):
            await self.manager.execute_tool("", {"param1": "value"})

if __name__ == '__main__':
    unittest.main()
