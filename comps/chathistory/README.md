# 📝 Chat History Microservice

The Chat History Microservice is a scalable solution for storing, retrieving and managing chat conversations using various type of databases. This microservice is designed to seamlessly integrate with OPEA chat applications, enabling data persistence and efficient management of chat histories.

It can be integrated into application by making HTTP requests to the provided API endpoints as shown in the flow diagram below.

![Flow Chart](./assets/img/chathistory_flow.png)

---

## 🛠️ Features

- **Store Chat Conversations**: Save chat messages user information, and metadata associated with each conversation.
- **Retrieve Chat Histories**: Fetch chat histories for a specific user or retrieve a particular conversation by its unique identifier.
- **Update Chat Conversations**: Modify existing chat conversations by adding new messages or updating existing ones.
- **Delete Chat Conversations**: Remove chat conversations record from database.

---

## 🤖 MCP (Model Context Protocol) Support

The Chat History microservice supports MCP integration, allowing AI agents to discover and utilize chat history management capabilities as tools.

### MCP Configuration

#### Environment Variables

- `ENABLE_MCP`: Set to `true`, `1`, or `yes` to enable MCP support (default: `false`)

#### Docker Compose

```yaml
services:
  chathistory-mongo:
    environment:
      ENABLE_MCP: true
```

#### Kubernetes

```yaml
chathistory:
  ENABLE_MCP: true
```

### MCP Tools Available

When MCP is enabled, the following tools are available for AI agents:

1. **create_documents** - Create or update chat conversation history
2. **get_documents** - Retrieve chat conversation history
3. **delete_documents** - Delete chat conversation history

### Usage with AI Agents

```python
from comps.cores.mcp import OpeaMCPToolsManager

# Initialize MCP tools manager
tools_manager = OpeaMCPToolsManager()

# Add chathistory service
tools_manager.add_service("http://chathistory-service:6012")

# AI agents can now discover and use chathistory tools
tools = await tools_manager.get_available_tools()
```

### MCP Endpoint

When MCP is enabled, the service exposes an additional SSE endpoint:

- `/sse` - Server-Sent Events endpoint for MCP communication

---

## ⚙️ Implementation

The Chat History microservice able to support various database backends for storing the chat conversations.

### Chat History with MongoDB

For more detail, please refer to this [README](src/README.md)
