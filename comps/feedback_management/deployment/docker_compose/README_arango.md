# 🗨 Feedback Management Microservice with ArangoDB

This README provides setup guides and all the necessary information about the Feedback Management microservice with ArangoDB database.

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
```

---

## 🚀 Start Microservice with Docker (Option 1)

### Build Docker Image

```bash
cd ~/GenAIComps
docker build -t opea/feedbackmanagement:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/feedback_management/src/Dockerfile .
```

### Run Docker with CLI

- Run ArangoDB image container

  ```bash
  docker run -d -p 8529:8529 --name=arango-vector-db -e ARANGO_ROOT_PASSWORD=${ARANGO_ROOT_PASSWORD} arangodb/arangodb:latest
  ```

- Run Feedback Management microservice

  ```bash
  docker run -d --name="feedbackmanagement-arango-server" -p 6016:6016 -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy -e ARANGODB_HOST=${ARANGODB_HOST} -e ARANGODB_USERNAME=${ARANGODB_USERNAME} -e ARANGODB_PASSWORD=${ARANGODB_PASSWORD} -e ARANGODB_DB_NAME=${ARANGODB_DB_NAME} -e ARANGODB_COLLECTION_NAME=${ARANGODB_COLLECTION_NAME} opea/feedbackmanagement:latest
  ```

---

## 🚀 Start Microservice with Docker Compose (Option 2)

```bash
docker compose -f ../deployment/docker_compose/compose.yaml up -d feedbackmanagement-arango
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


  # Take note that chat_id here would be the id get from chathistory_arango service
  # If you do not wish to maintain chat history via chathistory_arango service, you may generate some random uuid for it or just leave it empty.
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
