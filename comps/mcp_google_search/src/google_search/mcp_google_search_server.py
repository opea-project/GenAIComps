# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapt from https://github.com/modelcontextprotocol/quickstart-resources/blob/main/weather-server-python/src/weather/server.py

import os

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import asyncio

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.utilities import GoogleSearchAPIWrapper


server = Server("google-search")
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="get-google-search-answer",
            description="Get the google search retrieved documents that are relevant to a query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query from the user.",
                    },
                    "num_results": {
                        "type": "number",
                        "description": "How many urls to look up."
                    },
                    "google_api_key": {
                        "type": "string",
                        "description": "Google API KEY."
                    },
                    "google_cse_id": {
                        "type": "string",
                        "description": "Google CSE ID."
                    }
                },
                "required": ["query", "google_api_key", "google_cse_id"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can fetch google results based on the query.
    """
    if not arguments:
        raise ValueError("Missing arguments")
    if not arguments.get("google_api_key") or not arguments.get("google_cse_id"):
        raise ValueError("Missing environment variable GOOGLE_API_KEY or GOOGLE_CSE_ID")

    if name == "get-google-search-answer":
        try:
            query = arguments.get("query")
            num_results = arguments.get("num_results ", 4)
            if not query:
                raise ValueError("Missing query parameter")
            search = GoogleSearchAPIWrapper(
                    google_api_key=arguments.get("google_api_key"), google_cse_id=arguments.get("google_cse_id"), k=10
            )
            search_results = search.results(query, num_results)
            urls_to_look = []

            for res in search_results:
                if res.get("link", None):
                    urls_to_look.append(res["link"])
            urls = list(set(urls_to_look))
            loader = AsyncHtmlLoader(urls, ignore_load_errors=True, trust_env=True)
            docs = loader.load()
            html2text = Html2TextTransformer()
            docs = list(html2text.transform_documents(docs))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
            docs = text_splitter.split_documents(docs)
            # Remove duplicated docs
            unique_documents_dict = {(doc.page_content, tuple(sorted(doc.metadata.items()))): doc for doc in docs}
            # unique_documents = list(unique_documents_dict.values())
            page_content = [k[0] for k in unique_documents_dict.keys()]

            # return the retrieved docs as context directly
            return [
                    types.TextContent(
                        type="text",
                        text="\n".join(page_content)
                    )
                ]
        except Exception as e:
            return [
                    types.TextContent(
                        type="text",
                        text=str(e)
                    )
                ]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="google-search",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())