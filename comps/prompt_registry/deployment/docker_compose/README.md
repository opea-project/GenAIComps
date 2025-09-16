# 🧾 Prompt Registry Microservice with MongoDB

This README provides setup guides and all the necessary information about the Prompt Registry microservice with MongoDB database.

---

## Setup Environment Variables

```bash
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export OPEA_STORE_NAME="mongodb"
export MONGO_HOST=${MONGO_HOST}
export MONGO_HOST=27017
export DB_NAME=${DB_NAME}
export COLLECTION_NAME=${COLLECTION_NAME}
export ENABLE_MCP=false  # Set to true to enable MCP support
```

---

## 🚀 Start Microservice with Docker (Option 1)

### Build Docker Image

```bash
cd ~/GenAIComps
docker build -t opea/promptregistry:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/src/Dockerfile .
```

### Run Docker with CLI

- Run MongoDB image container

  ```bash
  docker run -d -p 27017:27017 --name=mongo mongo:latest
  ```

- Run Prompt Registry microservice

  ```bash
  docker run -d --name="promptregistry-mongo-server" -p 6018:6018 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e MONGO_HOST=${MONGO_HOST} -e MONGO_PORT=${MONGO_PORT} -e DB_NAME=${DB_NAME} -e COLLECTION_NAME=${COLLECTION_NAME} -e ENABLE_MCP=${ENABLE_MCP} opea/promptregistry:latest
  ```

---

## 🚀 Start Microservice with Docker Compose (Option 2)

```bash
docker compose -f ../deployment/docker_compose/compose.yaml up -d promptregistry-mongo
```

---

### ✅ Invoke Microservice

The Prompt Registry microservice exposes the following API endpoints:

- Save prompt

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6018/v1/prompt/create \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt_text": "test prompt", "user": "test"
  }'
  ```

- Retrieve prompt from database by user

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6018/v1/prompt/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test"}'
  ```

- Retrieve prompt from database by prompt_id

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6018/v1/prompt/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "prompt_id":"{prompt_id returned from save prompt route above}"}'
  ```

- Retrieve relevant prompt by keyword

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6018/v1/prompt/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "prompt_text": "{keyword to search}"}'
  ```

- Delete prompt by prompt_id

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6018/v1/prompt/delete \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "prompt_id":"{prompt_id to be deleted}"}'
  ```

---

## 🤖 MCP (Model Context Protocol) Usage

When MCP is enabled (`ENABLE_MCP=true`), the service operates in MCP-only mode. **Note: Regular HTTP endpoints are not available when MCP is enabled.** AI agents can discover and use the prompt registry service through the OPEA MCP framework:

### Available MCP Tools

- **create_prompt**: Store prompts with user association
- **get_prompt**: Retrieve prompts by various criteria
- **delete_prompt**: Remove prompts from the database

### Example MCP Client Usage

```python
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def use_prompt_registry():
    async with sse_client("http://localhost:6018/sse") as streams:
        async with ClientSession(*streams) as session:
            # Initialize connection
            await session.initialize()

            # Create a prompt
            result = await session.call_tool(
                "create_prompt",
                {"request": {"prompt_text": "Explain quantum computing in simple terms", "user": "ai_agent_001"}},
            )

            # Retrieve prompts
            prompts = await session.call_tool("get_prompt", {"request": {"user": "ai_agent_001"}})
```

### Benefits for AI Agents

- **Dynamic Prompt Management**: Agents can build prompt libraries on-the-fly
- **Context Persistence**: Store and retrieve prompts across conversations
- **Collaborative Learning**: Share effective prompts between agents
- **Personalization**: Create user-specific prompt repositories
