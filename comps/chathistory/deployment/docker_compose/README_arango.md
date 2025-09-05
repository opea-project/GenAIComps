# 📝 Chat History Microservice with ArangoDB

This README provides setup guides and all the necessary information about the Chat History microservice with ArangoDB database.

---

## Setup Environment Variables

```bash
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export OPEA_STORE_NAME="arangodb"
export ARANGODB_HOST="http://localhost:8529"
export ARANGODB_USERNAME="root"
export ARANGODB_PASSWORD="${YOUR_ARANGO_PASSWORD}"
export ARANGODB_ROOT_PASSWORD="${YOUR_ARANGO_ROOT_PASSWORD}"
export ARANGODB_DB_NAME="${YOUR_ARANGODB_DB_NAME-_system}"
export ARANGODB_COLLECTION_NAME="${YOUR_ARANGODB_COLLECTION_NAME-default}"
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

- Run ArangoDB image container

  ```bash
  docker run -d -p 8529:8529 --name=arango-vector-db -e ARANGO_ROOT_PASSWORD=${ARANGO_ROOT_PASSWORD} arangodb/arangodb:latest
  ```

- Run the Chat History microservice

  ```bash
  docker run -d --name="chathistory-arango-server" -p 6012:6012 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e ARANGODB_HOST=${ARANGODB_HOST} -e ARANGODB_USERNAME=${ARANGODB_USERNAME} -e ARANGODB_PASSWORD=${ARANGODB_PASSWORD} -e ARANGODB_DB_NAME=${ARANGODB_DB_NAME} -e ARANGODB_COLLECTION_NAME=${ARANGODB_COLLECTION_NAME} -e ENABLE_MCP=${ENABLE_MCP} opea/chathistory:latest
  ```

---

## 🚀 Start Microservice with Docker Compose (Option 2)

```bash
docker compose -f ../deployment/docker_compose/compose.yaml up -d chathistory-arango
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
