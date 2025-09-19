# 📝 Chat History Microservice with Redis

This README provides setup guides and all the necessary information about the Chat History microservice with Redis database.

---

## Setup Environment Variables

```bash
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export OPEA_STORE_NAME="redis"
export REDIS_URL="${REDIS_URL-redis://localhost:6379}"
export INDEX_NAME="${INDEX_NAME-opea:index}"
export DOC_PREFIX="${DOC_PREFIX-doc:}"
export AUTO_CREATE_INDEX="${AUTO_CREATE_INDEX-true}"
export ENABLE_MCP=false  # Set to true to enable MCP support
```

---

## 🚀 Start Microservice with Docker (Option 1)

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/chathistory:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/chathistory/src/Dockerfile .
```

### Run Docker with CLI

- Run Redis image container

  ```bash
  docker run -d -p 6379:6379 --name=redis-kv-store redis/redis-stack:latest
  ```

- Run the Chat History microservice

  ```bash
  docker run -d --name="chathistory-redis-server" -p 6012:6012 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e OPEA_STORE_NAME=redis -e REDIS_URL=${REDIS_URL} -e INDEX_NAME=${INDEX_NAME} -e DOC_PREFIX=${DOC_PREFIX} -e AUTO_CREATE_INDEX=${AUTO_CREATE_INDEX} -e ENABLE_MCP=${ENABLE_MCP} opea/chathistory:latest
  ```

---

## 🚀 Start Microservice with Docker Compose (Option 2)

```bash
docker compose -f ../deployment/docker_compose/compose.yaml up -d chathistory-redis
```

---

## ✅ Invoke Microservice

The Chat History microservice exposes the following API endpoints:

- Create new chat conversation

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6012/v1/chathistory/create \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "data": {
      "messages": "test Messages", "user": "test"
    }
  }'
  ```

- Get all the Conversations for a user

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6012/v1/chathistory/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test"}'
  ```

- Get a specific conversation by id.

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6012/v1/chathistory/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "id":"YOU_COLLECTION_NAME/YOU_DOC_KEY"}'
  ```

- Update the conversation by id.

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6012/v1/chathistory/create \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "data": {
      "messages": "test Messages Update", "user": "test"
    },
    "id":"YOU_COLLECTION_NAME/YOU_DOC_KEY"
  }'
  ```

- Delete a stored conversation.

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6012/v1/chathistory/delete \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "id":"YOU_COLLECTION_NAME/YOU_DOC_KEY"}'
  ```
