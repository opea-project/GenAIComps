# 🧾 Prompt Registry Microservice with ArangoDB

This README provides setup guides and all the necessary information about the Prompt Registry microservice with ArangoDB database.

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
cd ~/GenAIComps
docker build -t opea/promptregistry-arango-server:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/prompt_registry/arango/Dockerfile .
```

### Run Docker with CLI


- Run ArangoDB image container

  ```bash
  docker run -d -p 8529:8529 --name=arango arangodb/arangodb:latest
  ```

- Run Prompt Registry microservice

  ```bash
  docker run -d -p 6018:6018 \
  --name="promptregistry-arango-server" \  
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
  opea/promptregistry-arango-server:latest

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
    "user": "test", "prompt_id":"{_id returned from save prompt route above}"}'
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
