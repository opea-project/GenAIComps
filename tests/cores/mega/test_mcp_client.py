# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import Tool

from comps.cores.mcp.client import OpeaMCPClient
from comps.cores.mcp.config import OpeaMCPConfig, OpeaMCPSSEServerConfig, OpeaMCPStdioServerConfig
from comps.cores.mcp.manager import OpeaMCPToolsManager
from comps.cores.mcp.tool import OpeaMCPClientTool


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

    @patch.object(OpeaMCPToolsManager, "_initialize_clients", return_value=[])
    @patch.object(OpeaMCPToolsManager, "_register_tools")
    def test_create_success(self, mock_register_tools, mock_initialize_clients):
        """Test that OpeaMCPToolsManager can be successfully created."""
        asyncio.run(self.manager.create(self.mock_config))

        mock_initialize_clients.assert_called_once()
        mock_register_tools.assert_called_once()

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    @patch("comps.cores.mcp.manager.logger")
    async def test_initialize_clients_success(self, mock_logger, MockOpeaMCPClient):
        """Test the _initialize_clients method with valid configurations."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client

        # Mock successful connections
        mock_client.connect_via_sse = MagicMock()
        mock_client.connect_via_stdio = MagicMock()

        initialized_clients = await self.manager._initialize_clients([self.mock_sse_config, self.mock_stdio_config])

        # Assert that the client was initialized successfully
        self.assertEqual(len(initialized_clients), 2)
        mock_client.connect_via_sse.assert_called_once_with(self.mock_sse_config.url, self.mock_sse_config.api_key)
        mock_client.connect_via_stdio.assert_called_once_with(
            self.mock_stdio_config.command, self.mock_stdio_config.args
        )

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    @patch("comps.cores.mcp.manager.logger")
    async def test_initialize_clients_failure(self, mock_logger, MockOpeaMCPClient):
        """Test the _initialize_clients method when a client connection fails."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client

        # Simulate an error during connection
        mock_client.connect_via_sse.side_effect = Exception("Connection failed")

        initialized_clients = await self.manager._initialize_clients([self.mock_sse_config])

        # Assert that no clients were initialized due to the failure
        self.assertEqual(len(initialized_clients), 0)

    @patch.object(
        OpeaMCPToolsManager, "_extract_tools_from_clients", return_value=[{"name": "mock_tool", "params": {}}]
    )
    @patch("comps.cores.mcp.manager.logger")
    async def test_register_tools(self, mock_logger, mock_extract_tools):
        """Test tool registration."""
        await self.manager._register_tools()

        # Assert that _extract_tools_from_clients was called once
        mock_extract_tools.assert_called_once()
        # Check that a method has been dynamically added for the mock tool
        self.assertTrue(hasattr(self.manager, "mock_tool"))

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_execute_tool_tool_not_found(self, MockOpeaMCPClient):
        """Test executing a tool when the tool is not found in any client."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = []

        with self.assertRaises(ValueError) as context:
            await self.manager.execute_tool("mock_tool", {"param1": "value"})

        self.assertEqual(str(context.exception), "No MCP client found that provides the tool: mock_tool")

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_execute_tool_with_sse(self, MockOpeaMCPClient):
        """Test tool execution when using an SSE server."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.invoke_tool = MagicMock(return_value={"result": "sse_tool_success"})

        # Add mock tool for SSE
        mock_client.tools = [MagicMock(name="sse_tool", to_param=MagicMock(return_value={"name": "sse_tool"}))]

        result = await self.manager.execute_tool("sse_tool", {"param1": "value"})

        self.assertEqual(result, '{"result": "sse_tool_success"}')
        mock_client.invoke_tool.assert_called_once_with("sse_tool", {"param1": "value"})

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_context_manager_disconnects(self, MockOpeaMCPClient):
        """Test that the context manager disconnects all clients."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        self.manager.clients = [mock_client]

        async with self.manager:
            pass  # Test that __aenter__ works

        mock_client.disconnect.assert_called_once()

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_initialize_clients_with_failure(self, MockOpeaMCPClient):
        """Test that clients are handled correctly when initialization fails."""
        MockOpeaMCPClient.return_value.connect_via_sse = MagicMock(side_effect=Exception("Connection failed"))

        # This should result in an empty client list after failure
        clients = await self.manager._initialize_clients([self.mock_sse_config])
        self.assertEqual(clients, [])

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_initialize_clients_partial_failure(self, MockOpeaMCPClient):
        """Test that _initialize_clients handles partial failures correctly."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client

        # Simulate one successful connection and one failure
        mock_client.connect_via_sse.side_effect = [None, Exception("Connection failed")]

        clients = await self.manager._initialize_clients([self.mock_sse_config, self.mock_sse_config])

        # Assert that only one client was initialized successfully
        self.assertEqual(len(clients), 1)
        mock_client.connect_via_sse.assert_called()

    @patch.object(OpeaMCPToolsManager, "_extract_tools_from_clients", return_value=[])
    async def test_register_tools_no_tools(self, mock_extract_tools):
        """Test _register_tools when no tools are available."""
        await self.manager._register_tools()

        # Assert that no tools were registered
        mock_extract_tools.assert_called_once()
        self.assertEqual(len(self.manager.tools_registry), 0)

    @patch.object(
        OpeaMCPToolsManager,
        "_extract_tools_from_clients",
        return_value=[{"name": "duplicate_tool"}, {"name": "duplicate_tool"}],
    )
    async def test_register_tools_duplicate_tools(self, mock_extract_tools):
        """Test _register_tools with duplicate tool names."""
        await self.manager._register_tools()

        # Assert that duplicate tools are handled correctly
        self.assertEqual(len(self.manager.tools_registry), 2)
        self.assertTrue(hasattr(self.manager, "duplicate_tool"))

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_context_manager_with_exception(self, MockOpeaMCPClient):
        """Test that the context manager handles exceptions correctly."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        self.manager.clients = [mock_client]

        with self.assertRaises(Exception):
            async with self.manager:
                raise Exception("Test exception")

        # Assert that disconnect was called even after an exception
        mock_client.disconnect.assert_called_once()

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_execute_tool_invalid_parameters(self, MockOpeaMCPClient):
        """Test execute_tool with invalid parameters."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = [MagicMock(name="mock_tool")]

        with self.assertRaises(TypeError):
            await self.manager.execute_tool("mock_tool", None)

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_execute_tool_empty_tool_name(self, MockOpeaMCPClient):
        """Test execute_tool with an empty tool name."""
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = []

        with self.assertRaises(ValueError):
            await self.manager.execute_tool("", {"param1": "value"})

    def test_extract_tools_from_empty_clients(self):
        self.manager.clients = []
        result = self.manager._extract_tools_from_clients(self.manager.clients)
        self.assertEqual(result, [])

    def test_extract_tools_with_invalid_tool(self):
        mock_tool = MagicMock()
        mock_tool.to_param.side_effect = Exception("Invalid tool")
        mock_client = MagicMock()
        mock_client.tools = [mock_tool]
        self.manager.clients = [mock_client]

        result = self.manager._extract_tools_from_clients(self.manager.clients)
        self.assertEqual(result, [])

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_dynamic_tool_method_failure(self, MockOpeaMCPClient):
        mock_client = MagicMock()
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = [MagicMock(name="fail_tool", to_param=MagicMock(return_value={"name": "fail_tool"}))]
        mock_client.invoke_tool.side_effect = Exception("Invocation failed")

        self.manager.clients = [mock_client]
        await self.manager._register_tools()

        with self.assertRaises(Exception) as ctx:
            await getattr(self.manager, "fail_tool")({"some": "param"})
        self.assertIn("Invocation failed", str(ctx.exception))

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_dynamic_tool_method_failure(self, MockOpeaMCPClient):
        mock_client = MagicMock()
        MockOpeaMCPClient.return_value = mock_client
        mock_client.tools = [MagicMock(name="fail_tool", to_param=MagicMock(return_value={"name": "fail_tool"}))]
        mock_client.invoke_tool.side_effect = Exception("Invocation failed")

        self.manager.clients = [mock_client]
        await self.manager._register_tools()

        with self.assertRaises(Exception) as ctx:
            await getattr(self.manager, "fail_tool")({"some": "param"})
        self.assertIn("Invocation failed", str(ctx.exception))

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_context_manager_enter_returns_self(self, MockOpeaMCPClient):
        self.manager.clients = []
        async with self.manager as mgr:
            self.assertIs(mgr, self.manager)

    def test_extract_tools_from_clients(self):
        mock_tool = MagicMock()
        mock_tool.to_param.return_value = {"name": "test_tool"}
        mock_client = MagicMock()
        mock_client.tools = [mock_tool]

        self.manager.clients = [mock_client]
        tools = self.manager._extract_tools_from_clients(self.manager.clients)

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_tool")

    @patch.object(OpeaMCPToolsManager, "_extract_tools_from_clients", return_value=[{"name": "mock_tool"}])
    async def test_registered_tool_method_is_callable(self, mock_extract):
        await self.manager._register_tools()
        self.assertTrue(callable(getattr(self.manager, "mock_tool", None)))

    async def test_add_tool_method_with_missing_name(self):
        self.manager.clients = []
        tool_with_no_name = {"param": "value"}  # Missing 'name'
        with self.assertRaises(KeyError):
            self.manager._add_tool_method(tool_with_no_name)

    async def test_context_manager_enter_returns_self(self):
        async with self.manager as mgr:
            self.assertIs(mgr, self.manager)

    @patch.object(OpeaMCPToolsManager, "_extract_tools_from_clients")
    async def test_register_tools_conflict_resolution(self, mock_extract):
        # Two tools with same name, should both be added to registry
        mock_extract.return_value = [{"name": "conflict_tool"}, {"name": "conflict_tool"}]  # duplicate
        await self.manager._register_tools()

        # Confirm dynamic method is registered once, but registry has both entries
        self.assertTrue(hasattr(self.manager, "conflict_tool"))
        self.assertEqual(len(self.manager.tools_registry), 2)

    def test_extract_tools_from_clients_with_exception(self):
        broken_client = MagicMock()
        broken_client.tools = property(lambda _: (_ for _ in ()).throw(Exception("Tool access error")))
        manager = OpeaMCPToolsManager(config=MagicMock())
        tools = manager._extract_tools_from_clients([broken_client])
        self.assertEqual(tools, [])

    async def test_execute_tool_success(self):
        dummy_response = MagicMock()
        dummy_response.model_dump.return_value = {"result": "ok"}

        dummy_tool = MagicMock()
        dummy_tool.name = "my_tool"

        dummy_client = MagicMock()
        dummy_client.tools = [dummy_tool]
        dummy_client.invoke_tool = AsyncMock(return_value=dummy_response)

        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = [dummy_client]

        result = await manager.execute_tool("my_tool", {"x": 1})

        self.assertEqual(result, '{"result": "ok"}')

    async def test_async_exit_disconnects_clients(self):
        dummy_client = MagicMock()
        dummy_client.disconnect = AsyncMock()

        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = [dummy_client]

        await manager.__aexit__(None, None, None)
        dummy_client.disconnect.assert_called_once()

    async def test_async_exit_with_disconnect_error(self):
        dummy_client = MagicMock()
        dummy_client.disconnect = AsyncMock(side_effect=Exception("disconnect failed"))

        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = [dummy_client]

        with self.assertLogs("comps-mcp-manager", level="ERROR") as log:
            await manager.__aexit__(None, None, None)
            self.assertIn("Error while disconnecting MCP client", log.output[0])

    async def test_initialize_clients_sse(self):
        # Mock server configuration
        server_config = OpeaMCPSSEServerConfig(url="http://example.com", api_key="key123")
        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = []

        client_mock = AsyncMock()
        client_mock.connect_via_sse = AsyncMock(return_value=None)

        # Ensure the mock client is used
        OpeaMCPClient = MagicMock(return_value=client_mock)

        # Call _initialize_clients
        result = await manager._initialize_clients([server_config])

        # Check if client was initialized and connected via SSE
        client_mock.connect_via_sse.assert_called_with("http://example.com", "key123")
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], client_mock)

    async def test_initialize_clients_stdio(self):
        # Mock server configuration
        server_config = OpeaMCPStdioServerConfig(command="run_command", args=["arg1", "arg2"], name="test_name")
        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = []

        client_mock = AsyncMock()
        client_mock.connect_via_stdio = AsyncMock(return_value=None)
        OpeaMCPClient = MagicMock(return_value=client_mock)

        # Call _initialize_clients
        result = await manager._initialize_clients([server_config])

        # Check if client was initialized and connected via Stdio
        client_mock.connect_via_stdio.assert_called_with("run_command", ["arg1", "arg2"])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], client_mock)

    async def test_initialize_clients_error_handling(self):
        # Mock server configuration
        server_config = OpeaMCPSSEServerConfig(url="http://example.com", api_key="key123")
        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = []

        client_mock = AsyncMock()
        client_mock.connect_via_sse = AsyncMock(side_effect=Exception("Connection failed"))
        OpeaMCPClient = MagicMock(return_value=client_mock)

        # Call _initialize_clients
        result = await manager._initialize_clients([server_config])

        # Check that the client is not added to the list
        self.assertEqual(len(result), 0)

    async def test_register_tools_and_call(self):
        tool_name = "custom_tool"

        async def fake_exec(tool_name, params):
            return "executed"

        dummy_tool = {"function": {"name": tool_name}}

        manager = OpeaMCPToolsManager(config=MagicMock())
        manager._extract_tools_from_clients = MagicMock(return_value=[dummy_tool])
        manager.execute_tool = AsyncMock(side_effect=fake_exec)

        # Register the tools
        manager._register_tools()

        # Check if the tool method is dynamically added to the manager
        self.assertTrue(hasattr(manager, tool_name))

        # Call the dynamically added method
        result = await getattr(manager, tool_name)()
        self.assertEqual(result, "executed")

    async def test_execute_tool_no_clients(self):
        manager = OpeaMCPToolsManager(config=MagicMock())
        manager.clients = []  # No clients connected

        with self.assertRaises(ValueError) as context:
            await manager.execute_tool("custom_tool", {})

        self.assertEqual(str(context.exception), "No MCP clients are currently connected.")

    async def test_execute_tool_no_matching_tool(self):
        manager = OpeaMCPToolsManager(config=MagicMock())
        client_mock = MagicMock()
        client_mock.tools = []  # Client has no tools
        manager.clients = [client_mock]

        with self.assertRaises(ValueError) as context:
            await manager.execute_tool("custom_tool", {})

        self.assertEqual(str(context.exception), "No MCP client found that provides the tool: custom_tool")

    async def test_execute_tool_success(self):
        manager = OpeaMCPToolsManager(config=MagicMock())
        client_mock = MagicMock()
        client_mock.tools = [MagicMock(name="custom_tool")]
        manager.clients = [client_mock]

        client_mock.invoke_tool = AsyncMock(return_value={"result": "ok"})

        result = await manager.execute_tool("custom_tool", {})

        self.assertEqual(result, '{"result": "ok"}')


class TestOpeaMCPClient(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.client = OpeaMCPClient()

    @patch("comps.cores.mcp.client.ClientSession")
    @patch("comps.cores.mcp.client.sse_client")
    @patch("anyio.create_task_group")
    async def test_connect_via_sse_success(self, mock_create_task_group, mock_sse_client, mock_client_session):
        # Mock task group to avoid actual task creation
        mock_create_task_group.return_value = AsyncMock()

        # Mock the SSE client context manager to simulate a successful connection
        mock_sse_context = AsyncMock()
        mock_sse_context.__aenter__.return_value = ("reader", "writer")
        mock_sse_client.return_value = mock_sse_context

        # Mock ClientSession to simulate a valid session
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = mock_session
        mock_client_session.return_value = mock_session

        # Create the client instance
        client = OpeaMCPClient()

        # Set timeout for the test
        timeout = 10  # Reasonable timeout for testing

        # Run the connection method with mock objects
        await client.connect_via_sse("http://example.com", api_key="key", timeout=timeout)

        # Assertions
        self.assertIsNotNone(client.session)
        self.assertEqual(client.session, mock_session)

        # Ensure the mock session, sse_client, and task group were called as expected
        mock_session.__aenter__.assert_called_once()
        mock_sse_client.assert_called_once_with(
            url="http://example.com", headers={"Authorization": "Bearer key"}, timeout=10
        )

    async def test_connect_via_sse_missing_url(self):
        # Testing with an empty URL, expecting a ValueError
        with self.assertRaises(ValueError):
            await self.client.connect_via_sse("")

    @patch("mcp.ClientSession")
    @patch("mcp.client.sse.sse_client")
    @patch("comps.cores.mcp.client.OpeaMCPClient.disconnect", new_callable=AsyncMock)
    async def test_connect_via_sse_timeout(self, mock_disconnect, mock_sse_client, mock_client_session):
        mock_connect = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_sse_client.return_value.__aenter__.side_effect = mock_connect

        with self.assertRaises(asyncio.TimeoutError):
            await self.client.connect_via_sse("http://slow.server", timeout=0.01)

        mock_disconnect.assert_awaited()

    @patch("mcp.client.stdio._create_platform_compatible_process")
    @patch("mcp.client.stdio.stdio_client")
    @patch("comps.cores.mcp.client.ClientSession")
    async def test_connect_via_stdio_success(self, mock_client_session, mock_stdio_client, mock_create_process):
        mock_stdio = AsyncMock()
        mock_stdio.__aenter__.return_value = ("read", "write")
        mock_stdio_client.return_value = mock_stdio

        mock_session = MagicMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        self.client._initialize_tools = AsyncMock()

        await self.client.connect_via_stdio("dummy_cmd", ["--flag"])

        self.assertEqual(self.client.session, mock_session)
        self.client._initialize_tools.assert_awaited_once()

    async def test_connect_via_stdio_missing_command(self):
        # Testing with an empty command, expecting a ValueError
        with self.assertRaises(ValueError):
            await self.client.connect_via_stdio("", [])

    async def test_initialize_tools_success(self):
        tool1 = Tool(name="tool1", description="desc1", inputSchema={})
        tool2 = Tool(name="tool2", description="desc2", inputSchema={})

        mock_session = AsyncMock()
        mock_session.list_tools.return_value = SimpleNamespace(tools=[tool1, tool2])

        self.client.session = mock_session

        await self.client._initialize_tools()

        self.assertEqual(len(self.client.tools), 2)
        self.assertIn("tool1", self.client.tool_registry)
        self.assertIn("tool2", self.client.tool_registry)

    async def test_initialize_tools_without_session(self):
        # Testing with no session, expecting RuntimeError
        self.client.session = None
        with self.assertRaises(RuntimeError):
            await self.client._initialize_tools()

    async def test_invoke_tool_success(self):
        # Mocking a successful tool invocation
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = {"result": "ok"}

        tool = OpeaMCPClientTool(name="test_tool", description="desc", inputSchema={}, session=mock_session)
        self.client.tool_registry["test_tool"] = tool
        self.client.session = mock_session

        result = await self.client.invoke_tool("test_tool", {"param": 1})
        self.assertEqual(result, {"result": "ok"})

    async def test_invoke_tool_not_found(self):
        # Testing tool not found scenario
        self.client.tool_registry = {}
        with self.assertRaises(ValueError):
            await self.client.invoke_tool("nonexistent_tool", {})

    async def test_invoke_tool_no_session(self):
        # Testing when no session is available
        self.client.tool_registry["tool"] = MagicMock()
        self.client.session = None
        with self.assertRaises(RuntimeError):
            await self.client.invoke_tool("tool", {})

    async def test_disconnect_success(self):
        # Testing successful disconnection
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        self.client.session = mock_session
        self.client.exit_stack = AsyncMock()

        await self.client.disconnect()

        self.assertIsNone(self.client.session)
        self.assertEqual(self.client.tools, [])
        self.assertEqual(self.client.tool_registry, {})


class TestOpeaMCPConfig(unittest.TestCase):

    def test_valid_config_should_pass(self):
        config = OpeaMCPConfig(
            sse_servers=[
                OpeaMCPSSEServerConfig(url="http://example.com", api_key="abc123"),
                OpeaMCPSSEServerConfig(url="https://another.com", api_key=None),
            ],
            stdio_servers=[OpeaMCPStdioServerConfig(name="local", command="run_server", args=["--port", "8080"])],
        )

        # Should not raise
        config.validate_servers()
        self.assertEqual(len(config.sse_servers), 2)
        self.assertEqual(config.sse_servers[0].api_key, "abc123")
        self.assertEqual(config.stdio_servers[0].name, "local")

    def test_invalid_url_should_raise(self):
        config = OpeaMCPConfig(
            sse_servers=[
                OpeaMCPSSEServerConfig(url="invalid-url", api_key="xyz"),
            ]
        )
        with self.assertRaises(ValueError) as cm:
            config.validate_servers()
        self.assertIn("Invalid URL format", str(cm.exception))

    def test_duplicate_urls_should_raise(self):
        config = OpeaMCPConfig(
            sse_servers=[
                OpeaMCPSSEServerConfig(url="http://example.com", api_key="key1"),
                OpeaMCPSSEServerConfig(url="http://example.com", api_key="key2"),
            ]
        )
        with self.assertRaises(ValueError) as cm:
            config.validate_servers()
        self.assertIn("Duplicate MCP server URLs are not allowed", str(cm.exception))

    def test_empty_config_defaults(self):
        config = OpeaMCPConfig()
        self.assertEqual(config.sse_servers, [])
        self.assertEqual(config.stdio_servers, [])

    def test_stdio_server_env_defaults(self):
        server = OpeaMCPStdioServerConfig(name="stdio", command="cmd")
        self.assertEqual(server.args, [])
        self.assertEqual(server.env, {})


if __name__ == "__main__":
    unittest.main()
