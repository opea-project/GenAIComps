# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from comps.cores.mcp.client import OpeaMCPClient
from comps.cores.mcp.config import (
    OpeaMCPConfig,
    OpeaMCPSSEServerConfig,
    OpeaMCPStdioServerConfig,
)
from comps.cores.mcp.manager import OpeaMCPToolsManager
from comps.cores.mcp.tool import OpeaMCPClientTool


class TestOpeaMCPClientTool(unittest.TestCase):
    def test_to_param(self):
        tool = OpeaMCPClientTool(
            name="test_tool",
            description="desc",
            inputSchema={"type": "object"},
        )
        param = tool.to_param()
        self.assertEqual(param["function"]["name"], "test_tool")
        self.assertEqual(param["function"]["description"], "desc")
        self.assertEqual(param["function"]["parameters"], {"type": "object"})


class TestOpeaMCPConfig(unittest.TestCase):
    def test_config_init(self):
        sse = OpeaMCPSSEServerConfig(url="http://a", api_key="k")
        stdio = OpeaMCPStdioServerConfig(name="n", command="c", args=["a"])
        config = OpeaMCPConfig(sse_servers=[sse], stdio_servers=[stdio])
        self.assertEqual(config.sse_servers[0].url, "http://a")
        self.assertEqual(config.stdio_servers[0].name, "n")


class TestOpeaMCPClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = OpeaMCPClient()

    @patch("comps.cores.mcp.client.ClientSession")
    async def test_connect_via_sse_success(self, mock_session):
        # Patch sse_client context manager
        with patch("comps.cores.mcp.client.sse_client") as mock_sse_client:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = ("reader", "writer")
            mock_sse_client.return_value = mock_cm
            self.client._initialize_tools = AsyncMock()
            await self.client.connect_via_sse("http://test", api_key="k", timeout=0.1)
            self.client._initialize_tools.assert_awaited()

    @patch("comps.cores.mcp.client.ClientSession")
    async def test_connect_via_stdio_success(self, mock_session):
        with patch("comps.cores.mcp.client.stdio_client") as mock_stdio_client:
            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = ("reader", "writer")
            mock_stdio_client.return_value = mock_cm
            self.client._initialize_tools = AsyncMock()
            await self.client.connect_via_stdio("cmd", ["a"])
            self.client._initialize_tools.assert_awaited()

    async def test_initialize_tools(self):
        tool_data = MagicMock()
        tool1 = MagicMock()
        tool1.name = "t1"
        tool1.description = "d1"
        tool1.inputSchema = {"type": "object"}
        tool2 = MagicMock()
        tool2.name = "t2"
        tool2.description = "d2"
        tool2.inputSchema = {"type": "object"}
        tool_data.tools = [tool1, tool2]
        self.client.session = MagicMock()
        self.client.session.list_tools = AsyncMock(return_value=tool_data)
        self.client.session.initialize = AsyncMock()
        await self.client._initialize_tools()
        self.assertEqual(len(self.client.tools), 2)


class TestOpeaMCPToolsManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sse = OpeaMCPSSEServerConfig(url="http://a", api_key="k")
        self.stdio = OpeaMCPStdioServerConfig(name="n", command="c", args=["a"])
        self.config = OpeaMCPConfig(sse_servers=[self.sse], stdio_servers=[self.stdio])
        self.manager = OpeaMCPToolsManager(self.config)

    @patch.object(OpeaMCPToolsManager, "_initialize_clients", new_callable=AsyncMock)
    @patch.object(OpeaMCPToolsManager, "_register_tools", new_callable=AsyncMock)
    async def test_create(self, mock_register, mock_init_clients):
        await OpeaMCPToolsManager.create(self.config)
        mock_init_clients.assert_awaited()
        mock_register.assert_awaited()

    @patch("comps.cores.mcp.manager.OpeaMCPClient")
    async def test_initialize_clients(self, MockClient):
        mock_client = MagicMock(spec=OpeaMCPClient)
        MockClient.return_value = mock_client
        mock_client.connect_via_sse = AsyncMock()
        mock_client.connect_via_stdio = AsyncMock()
        clients = await self.manager._initialize_clients([self.sse, self.stdio])
        self.assertEqual(len(clients), 2)

    async def test_extract_tools_from_clients(self):
        mock_tool = MagicMock()
        mock_tool.to_param.return_value = {"function": {"name": "t"}}
        mock_client = MagicMock()
        mock_client.tools = [mock_tool]
        tools = self.manager._extract_tools_from_clients([mock_client])
        self.assertEqual(len(tools), 1)

    async def test_register_tools(self):
        self.manager.clients = []
        self.manager._extract_tools_from_clients = MagicMock(return_value=[{"function": {"name": "t"}}])
        self.manager.execute_tool = AsyncMock(return_value="ok")
        await self.manager._register_tools()
        self.assertTrue(hasattr(self.manager, "t"))
        result = await getattr(self.manager, "t")()
        self.assertEqual(result, "ok")

    async def test_execute_tool(self):
        dummy_tool = MagicMock()
        dummy_tool.name = "t"
        dummy_client = MagicMock()
        dummy_client.tools = [dummy_tool]
        dummy_client.invoke_tool = AsyncMock(return_value=MagicMock(model_dump=lambda mode: {"r": 1}))
        self.manager.clients = [dummy_client]
        result = await self.manager.execute_tool("t", {})
        self.assertEqual(result, '{"r": 1}')

    async def test_execute_tool_no_clients(self):
        self.manager.clients = []
        with self.assertRaises(ValueError):
            await self.manager.execute_tool("t", {})

    async def test_execute_tool_no_tool(self):
        dummy_client = MagicMock()
        dummy_client.tools = []
        self.manager.clients = [dummy_client]
        with self.assertRaises(ValueError):
            await self.manager.execute_tool("t", {})

    async def test_context_manager(self):
        dummy_client = MagicMock()
        dummy_client.disconnect = AsyncMock()
        self.manager.clients = [dummy_client]
        async with self.manager:
            pass
        dummy_client.disconnect.assert_awaited()

    async def test_context_manager_with_error(self):
        dummy_client = MagicMock()
        dummy_client.disconnect = AsyncMock(side_effect=Exception("fail"))
        self.manager.clients = [dummy_client]
        with self.assertLogs("comps-mcp-manager", level="ERROR"):
            await self.manager.__aexit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
