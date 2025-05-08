# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mcp.types import Tool


class OpeaMCPClientTool(Tool):
    """Represents a MCP tool proxy that can be called on the MCP server from the client side."""

    class Config:
        arbitrary_types_allowed = True

    def to_param(self) -> dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.inputSchema,
            },
        }
