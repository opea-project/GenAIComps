# 📝 Chat History Microservice with ArangoDB

This README provides setup guides and all the necessary information about the Chat History microservice with ArangoDB database.

---

## Setup Environment Variables

See `config.py` for default values.

```bash
export ARANGO_HOST=${ARANGO_HOST}
export ARANGO_PORT=${ARANGO_PORT}
export ARANGO_PROTOCOL=${ARANGO_PROTOCOL}
export ARANGO_USERNAME=${ARANGO_USERNAME}
export ARANGO_PASSWORD=${ARANGO_PASSWORD}
export DB_NAME=${DB_NAME}
export COLLECTION_NAME=${COLLECTION_NAME}
```

---

## 🚀Start Microservice with Docker

### Build Docker Image

```bash
cd ../../../../
docker build -t opea/chathistory-arango-server:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/chathistory/arango/Dockerfile .
```

### Run Docker with CLI

- Run ArangoDB image container

  ```bash
  docker run -d -p 8529:8529 --name=arango arangodb/arangodb:latest
  ```

- Run the Chat History microservice

  ```bash
  docker run -p 6012:6012 \  
  -e http_proxy=$http_proxy \
  -e https_proxy=$https_proxy \
  -e no_proxy=$no_proxy \
  -e ARANGO_HOST=${ARANGO_HOST} \
  -e ARANGO_PORT=${ARANGO_PORT} \
  -e ARANGO_PROTOCOL=${ARANGO_PROTOCOL} \
  -e ARANGO_USERNAME=${ARANGO_USERNAME} \
  -e ARANGO_PASSWORD=${ARANGO_PASSWORD} \
  -e DB_NAME=${DB_NAME} \
  -e COLLECTION_NAME=${COLLECTION_NAME} \
  opea/chathistory-arango-server:latest
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
    "user": "test", "id":"48918"}'
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
    "id":"48918"
  }'
  ```

- Delete a stored conversation.

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6012/v1/chathistory/delete \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "id":"48918"}'
  ```
