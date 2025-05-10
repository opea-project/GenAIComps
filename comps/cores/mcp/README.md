# OPEA MCP Tool

The **OPEA MCP Tool** is a client tool designed to facilitate seamless integration between agents and MCP (Model Context Protocol) servers. It provides a unified interface for managing MCP clients, enabling agents to access and interact with various tools and data sources exposed by MCP servers.

---

## **OPEA MCP Tool Overview**

The **OPEA MCP Tool** provides a unified interface for managing MCP clients and interacting with tools exposed by MCP servers. It supports both **SSE (Server-Sent Events)** and **Stdio** server configurations, making it flexible for various use cases.

### **Features**

- **Dynamic Tool Registration**: Automatically registers tools exposed by MCP servers for natural invocation.
- **Asynchronous Operations**: Fully asynchronous API for efficient integration with modern Python applications.
- **Context Management**: Supports Python's `async with` syntax for automatic resource management.
- **Error Handling**: Robust error handling for client initialization, tool execution, and disconnection.

---

## **API Usage**

### Initialization

To initialize the OpeaMCPToolsManager, provide an OpeaMCPConfig object containing the server configurations:

```python
from comps.cores.mcp.config import OpeaMCPConfig, OpeaMCPSSEServerConfig, OpeaMCPStdioServerConfig
from comps.cores.mcp.manager import OpeaMCPToolsManager

config = OpeaMCPConfig(
    sse_servers=[
        OpeaMCPSSEServerConfig(url="http://sse-server-1.com", api_key="your_api_key"),
    ],
    stdio_servers=[
        OpeaMCPStdioServerConfig(name="stdio-server-1", command="python", args=["tool.py"]),
    ],
)

manager = await OpeaMCPToolsManager.create(config)
```

### Tool Execution

Once initialized, you can execute tools exposed by MCP servers using the execute_tool method:

```python
result = await manager.execute_tool("tool_name", {"param1": "value1", "param2": "value2"})
print(result)
```

### Context Management

The OpeaMCPToolsManager supports Python's async with syntax for automatic resource management:

```python
async with await OpeaMCPToolsManager.create(config) as manager:
    result = await manager.execute_tool("tool_name", {"param1": "value1"})
    print(result)
```

### Dynamic Tool Invocation

Tools are dynamically registered as methods of the manager, allowing for natural invocation:

```python
async with OpeaMCPToolsManager.create(config) as manager:
    result = await manager.tool_name(param1="value1", param2="value2")
    print(result)
```

## **Examples**

### **Launch a SSE MCP Server**

To launch an SSE MCP server using Playwright, run the following command:

```bash
npx @playwright/mcp@latest --port 8931
```

### **Launch a Stdio MCP Server**

To launch a simple Stdio MCP server, follow these steps:

```bash
git clone https://github.com/modelcontextprotocol/python-sdk.git
cd python-sdk/examples/servers/simple-tool/mcp_simple_tool
uv run mcp-simple-tool
```

### **Run the MCP Client**

The following example demonstrates how to connect to both SSE and Stdio MCP servers and execute tools:

```python
import asyncio
from comps.cores.mcp.config import OpeaMCPConfig, OpeaMCPSSEServerConfig, OpeaMCPStdioServerConfig
from comps.cores.mcp.manager import OpeaMCPToolsManager


async def main():
    config = OpeaMCPConfig(
        sse_servers=[
            OpeaMCPSSEServerConfig(url="http://localhost:8931/sse"),
        ],
        stdio_servers=[
            OpeaMCPStdioServerConfig(name="mcp-simple-tool", command="uv", args=["run", "mcp-simple-tool"]),
        ],
    )

    async with await OpeaMCPToolsManager.create(config) as manager:
        # Execute tools exposed by the servers
        result = await manager.execute_tool("browser_snapshot", {})
        print(result)

        result = await manager.execute_tool("fetch", {"url": "https://opea.dev/"})
        print(result)


# Run the async function
asyncio.run(main())
```
