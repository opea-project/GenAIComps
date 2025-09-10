# 🗨 Feedback Management Microservice with Redis

This README provides setup guides and all the necessary information about the Feedback Management microservice with Redis database.

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
cd ~/GenAIComps
docker build -t opea/feedbackmanagement:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/feedback_management/src/Dockerfile .
```

### Run Docker with CLI

- Run Redis image container

  ```bash
  docker run -d -p 6379:6379 --name=redis-kv-store redis/redis-stack:latest
  ```

- Run Feedback Management microservice

  ```bash
  docker run -d --name="feedbackmanagement-redis-server" -p 6016:6016 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e OPEA_STORE_NAME=redis -e REDIS_URL=${REDIS_URL} -e INDEX_NAME=${INDEX_NAME} -e DOC_PREFIX=${DOC_PREFIX} -e AUTO_CREATE_INDEX=${AUTO_CREATE_INDEX} -e ENABLE_MCP=${ENABLE_MCP}  opea/feedbackmanagement:latest
  ```

---

## 🚀 Start Microservice with Docker Compose (Option 2)

```bash
docker compose -f ../deployment/docker_compose/compose.yaml up -d feedbackmanagement-redis
```

---

### ✅ Invoke Microservice

The Feedback Management microservice exposes the following API endpoints:

- Save feedback data

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6016/v1/feedback/create \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "chat_id": "66445d4f71c7eff23d44f78d",
    "chat_data": {
      "user": "test",
      "messages": [
        {
          "role": "system",
          "content": "You are helpful assistant"
        },
        {
          "role": "user",
          "content": "hi",
          "time": "1724915247"
        },
        {
          "role": "assistant",
          "content": "Hi, may I help you?",
          "time": "1724915249"
        }
      ]
    },
    "feedback_data": {
      "comment": "Moderate",
      "rating": 3,
      "is_thumbs_up": true
    }}'


  # Take note that chat_id here would be the id get from chathistory_redis service
  # If you do not wish to maintain chat history via chathistory_redis service, you may generate some random uuid for it or just leave it empty.
  ```

- Update feedback data by feedback_id

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6016/v1/feedback/create \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "chat_id": "66445d4f71c7eff23d44f78d",
    "chat_data": {
      "user": "test",
      "messages": [
        {
          "role": "system",
          "content": "You are helpful assistant"
        },
        {
          "role": "user",
          "content": "hi",
          "time": "1724915247"
        },
        {
          "role": "assistant",
          "content": "Hi, may I help you?",
          "time": "1724915249"
        }
      ]
    },
    "feedback_data": {
      "comment": "Fair and Moderate answer",
      "rating": 2,
      "is_thumbs_up": true
    },
    "feedback_id": "{feedback_id of the data that wanted to update}"}'

  # Just include any feedback_data field value that you wanted to update.
  ```

- Retrieve feedback data by user

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6016/v1/feedback/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test"}'
  ```

- Retrieve feedback data by feedback_id

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6016/v1/feedback/get \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "feedback_id":"{feedback_id returned from save feedback route above}"}'
  ```

- Delete feedback data by feedback_id

  ```bash
  curl -X 'POST' \
    http://${host_ip}:6016/v1/feedback/delete \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "user": "test", "feedback_id":"{feedback_id to be deleted}"}'
  ```
