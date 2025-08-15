# 🗨 Feedback Management Microservice

The Feedback Management microservice facilitates the storage and retrieval of users'feedback data by establishing a connection with the databases. This microservice is designed to seamlessly integrate with OPEA applications, enabling data persistence and efficient management of feedback data.

---

## 🛠️ Features

- **Store Feedback**: Save feedback data from user into database.
- **Retrieve Feedback**: Fetch feedback data from database based on user or id.
- **Update Feedback**: Update feedback data info in the database based on id.
- **Delete Feedback**: Remove feedback record from database.
- **MCP Support**: Enable AI agents to programmatically interact with feedback data through Model Context Protocol.

---

## ⚙️ Implementation

The Feedback Management microservice able to support various database backends for storing the feedback data.

### Feedback Management with MongoDB

For more detail, please refer to this [README](./src/README.md)

## 🤖 MCP (Model Context Protocol) Support

The Feedback Management microservice supports MCP, allowing AI agents to discover and use its functionality programmatically.

### Enabling MCP

To enable MCP support, set the environment variable:

```bash
export ENABLE_MCP=true
```

Or in your docker-compose.yaml:

```yaml
environment:
  ENABLE_MCP: true
```

### MCP Tools Available

When MCP is enabled, the following tools are exposed to AI agents:

1. **create_feedback_data** - Create or update feedback data for AI-generated responses including ratings and comments
2. **get_feedback** - Retrieve feedback data by ID or get all feedback for a specific user
3. **delete_feedback** - Delete specific feedback data by user ID and feedback ID

### Using with AI Agents

AI agents can connect to the service via the SSE transport endpoint at `/sse` when MCP is enabled. The service will be automatically discovered by agents using the OPEA MCP Tools Manager.

### Backward Compatibility

MCP support is disabled by default to maintain backward compatibility. The service continues to work normally via HTTP endpoints regardless of the MCP setting.
