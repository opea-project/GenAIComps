# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import unittest

from fastmcp import Client

from comps import TextDoc, opea_microservices, register_microservice
from comps.cores.mega.constants import MCPFuncType
from comps.version import __version__


@register_microservice(
    name="mcp_dummy",
    host="0.0.0.0",
    port=8087,
    # endpoint="/v1/add",
    enable_mcp=True,
    mcp_func_type=MCPFuncType.TOOL,
    description="dummy mcp add func",
)
async def mcp_dummy(request: TextDoc) -> TextDoc:
    req = request.model_dump_json()
    req_dict = json.loads(req)
    text = req_dict["text"]
    text += "OPEA Project MCP!"
    return {"text": text}


@register_microservice(
    name="mcp_dummy",
    host="0.0.0.0",
    port=8087,
    # endpoint="/v1/add",
    enable_mcp=True,
    mcp_func_type=MCPFuncType.TOOL,
    description="dummy mcp sum func",
)
async def mcp_dummy_sum():
    return 1 + 1


class TestMicroService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.process = multiprocessing.Process(
            target=opea_microservices["mcp_dummy"].start, daemon=False, name="mcp_dummy"
        )
        self.process.start()

        self.mcp_client = Client("http://localhost:8087/sse")

    def tearDown(self):
        self.process.terminate()

    async def test_mcp(self):

        async with self.mcp_client:
            self.assertTrue(self.mcp_client.is_connected())

            tools = await self.mcp_client.list_tools()
            self.assertEqual(tools[0].name, "mcp_dummy")
            result = await self.mcp_client.call_tool(
                "mcp_dummy",
                {"request": {"text": "Hello "}},
            )
            self.assertEqual(json.loads(result[0].text)["text"], "Hello OPEA Project MCP!")

            result = await self.mcp_client.call_tool(
                "mcp_dummy_sum",
            )
            self.assertEqual(result[0].text, "2")

        self.assertFalse(self.mcp_client.is_connected())


if __name__ == "__main__":
    unittest.main()
