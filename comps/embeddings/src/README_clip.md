# Multimodal CLIP Embedding Microservice

The Multimodal CLIP Embedding Microservice provides a powerful solution for converting textual and visual data into high-dimensional vector embeddings. These embeddings capture the semantic essence of the input, enabling robust applications in multi-modal data processing, information retrieval, recommendation systems, and more.

## ✨ Key Features

- **High Performance**: Optimized for rapid and reliable embedding generation for text and images.
- **Scalable**: Capable of handling high-concurrency workloads, ensuring consistent performance under heavy loads.
- **Easy Integration**: Offers a simple API interface for seamless integration into diverse workflows.
- **Customizable**: Supports tailored configurations, including model selection and preprocessing adjustments, to fit specific requirements.

This service empowers users to configure and deploy embedding pipelines tailored to their needs.

---

## 🚀 Quick Start

### 1. Launch the Microservice with Docker

#### 1.1 Build the Docker Image

To build the Docker image, execute the following commands:

```bash
cd ../../..
docker build -t opea/embedding:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f comps/embeddings/src/Dockerfile .
```

#### 1.2 Start the Service with Docker Compose

Use Docker Compose to start the service:

```bash
cd comps/embeddings/deployment/docker_compose/
docker compose up clip-embedding-server -d
```

---

### 2. Consume the Embedding Service

#### 2.1 Check Service Health

Verify that the service is running by performing a health check:

```bash
curl http://localhost:6000/v1/health_check \
  -X GET \
  -H 'Content-Type: application/json'
```

#### 2.2 Generate Embeddings

The service supports [OpenAI API](https://platform.openai.com/docs/api-reference/embeddings)-compatible requests.

- **Single Text Input**:

  ```bash
  curl http://localhost:6000/v1/embeddings \
    -X POST \
    -d '{"input":"Hello, world!"}' \
    -H 'Content-Type: application/json'
  ```

- **Multiple Texts with Parameters**:

  ```bash
  curl http://localhost:6000/v1/embeddings \
    -X POST \
    -d '{"input":["Hello, world!","How are you?"], "dimensions":100}' \
    -H 'Content-Type: application/json'
  ```
