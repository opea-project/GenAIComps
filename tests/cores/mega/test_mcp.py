# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import unittest

from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

from comps import TextDoc, opea_microservices, register_microservice
from comps.cores.mega.constants import MCPFuncType
from comps.version import __version__

@register_microservice(
    name="mcp_dummy",
    host="0.0.0.0",
    port=8087,
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

        self.server_url = "http://localhost:8087"


    async def test_mcp(self):
        async with sse_client(self.server_url + "/sse") as streams:
            async with ClientSession(*streams) as session:
                result = await session.initialize()
                self.assertEqual(result.serverInfo.name, "mcp_dummy")
                tool_result = await session.call_tool(
                    "mcp_dummy",
                    {"request": {"text": "Hello "}}
                )
                self.assertEqual(json.loads(tool_result.content[0].text)['text'], "Hello OPEA Project MCP!")
                tool_result = await session.call_tool(
                    "mcp_dummy_sum",
                )
                self.assertEqual(tool_result.content[0].text, "2")
            self.process.kill()
            self.process.join(timeout=2)

if __name__ == "__main__":
    unittest.main()
