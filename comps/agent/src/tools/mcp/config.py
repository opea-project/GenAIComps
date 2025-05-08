# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlparse

from pydantic import BaseModel, Field


class OpeaMCPSSEServerConfig(BaseModel):
    """Configuration for a single MCP server.

    Attributes:
        url: The server URL
        api_key: Optional API key for authentication
    """

    url: str
    api_key: str | None = None


class OpeaMCPStdioServerConfig(BaseModel):
    """Configuration for a MCP (Model Context Protocol) server that uses stdio.

    Attributes:
        name: The name of the server
        command: The command to run the server
        args: The arguments to pass to the server
        env: The environment variables to set for the server
    """

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class OpeaMCPConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) settings.

    Attributes:
        sse_servers: List of MCP SSE server configs
        stdio_servers: List of MCP stdio server configs. These servers will be added to the MCP Router running inside runtime container.
    """

    sse_servers: list[OpeaMCPSSEServerConfig] = Field(default_factory=list)
    stdio_servers: list[OpeaMCPStdioServerConfig] = Field(default_factory=list)

    def validate_servers(self) -> None:
        """Validate that server URLs are valid and unique."""
        urls = [server.url for server in self.sse_servers]

        # Check for duplicate server URLs
        if len(set(urls)) != len(urls):
            raise ValueError("Duplicate MCP server URLs are not allowed")

        # Validate URLs
        for url in urls:
            try:
                result = urlparse(url)
                if not all([result.scheme, result.netloc]):
                    raise ValueError(f"Invalid URL format: {url}")
            except Exception as e:
                raise ValueError(f"Invalid URL {url}: {str(e)}")
