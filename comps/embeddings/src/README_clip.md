# Multimodal CLIP Embedding Microservice

The Multimodal CLIP Embedding Microservice provides a powerful solution for converting textual and visual data into high-dimensional vector embeddings. These embeddings capture the semantic essence of the input, enabling robust applications in multi-modal data processing, information retrieval, recommendation systems, and more.

---

## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume Embedding Service](#consume-embedding-service)

---

## Start Microservice

### Build Docker Image

To build the Docker image, execute the following commands:

```bash
cd ../../..
docker build -t opea/embedding:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f comps/embeddings/src/Dockerfile .
```

### Run Docker with Docker Compose

```bash
cd comps/embeddings/deployment/docker_compose/
docker compose up clip-embedding-server -d
```

---

## Consume Embedding Service

### Check Service Status

Verify that the embedding service is running properly by checking its health status with this command:

```bash
curl http://localhost:6000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

### Use the Embedding Service API

The service supports [OpenAI API](https://platform.openai.com/docs/api-reference/embeddings)-compatible requests.

**Single Text Input**:

```bash
curl http://localhost:6000/v1/embeddings \
  -X POST \
  -d '{"input":"Hello, world!"}' \
  -H 'Content-Type: application/json'
```

**Multiple Texts with Parameters**:

```bash
curl http://localhost:6000/v1/embeddings \
  -X POST \
  -d '{"input":["Hello, world!","How are you?"], "dimensions":100}' \
  -H 'Content-Type: application/json'
```
