# 🧾 Prompt Registry Microservice

The Prompt Registry microservice facilitates the storage and retrieval of users' preferred prompts by establishing a connection with the databases. This microservice is designed to seamlessly integrate with OPEA applications, enabling data persistence and efficient management of user's preferred prompts.

---

## 🛠️ Features

- **Store Prompt**: Save user's preferred prompt into database.
- **Retrieve Prompt**: Fetch prompt from database based on user, id or even a keyword search.
- **Delete Prompt**: Remove prompt from database.
- **MCP Support**: Enable AI agents to discover and use prompt management capabilities via Model Context Protocol.

---

## ⚙️ Implementation

The Prompt Registry microservice able to support various database backends for storing the prompts.

### Prompt Registry with MongoDB

For more detail, please refer to this [README](./src/README.md)

### MCP (Model Context Protocol) Support

The Prompt Registry microservice supports MCP, allowing AI agents to discover and utilize its prompt management capabilities. When MCP is enabled, the service exposes three tools:

- `create_prompt`: Store a user's preferred prompt in the database
- `get_prompt`: Retrieve prompts by user, ID, or keyword search
- `delete_prompt`: Delete a prompt by ID from the database

To enable MCP support, set the `ENABLE_MCP` environment variable:

```bash
export ENABLE_MCP=true
```

Or in Docker Compose:

```yaml
environment:
  ENABLE_MCP: true
```

**Important Note**: When MCP is enabled (`ENABLE_MCP=true`), the service operates in MCP-only mode:

- Regular HTTP endpoints (`/v1/prompt/create`, `/v1/prompt/get`, `/v1/prompt/delete`) are not available
- The service only exposes the SSE endpoint (`/sse`) for MCP protocol communication
- AI agents interact with the service through MCP tools, not HTTP APIs
- To use both HTTP endpoints and MCP, you would need to run two instances of the service (one with MCP enabled, one without)

When MCP is enabled, AI agents can:

- Build and manage prompt libraries dynamically
- Reuse prompts across conversations
- Create personalized prompt repositories
- Share and discover prompts programmatically
