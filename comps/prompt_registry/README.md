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

- `opea_service@prompt_create`: Store a user's preferred prompt in the database
- `opea_service@prompt_get`: Retrieve prompts by user, ID, or keyword search
- `opea_service@prompt_delete`: Delete a prompt by ID from the database

To enable MCP support, set the `ENABLE_MCP` environment variable:

```bash
export ENABLE_MCP=true
```

Or in Docker Compose:

```yaml
environment:
  ENABLE_MCP: true
```

When MCP is enabled, AI agents can:

- Build and manage prompt libraries dynamically
- Reuse prompts across conversations
- Create personalized prompt repositories
- Share and discover prompts programmatically
