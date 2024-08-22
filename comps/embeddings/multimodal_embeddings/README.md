# Multimodal Embeddings Microservice

The Multimodal Embedding Microservice is designed to efficiently convert textual strings and images into vectorized embeddings, facilitating seamless integration into various machine learning and data processing workflows. This service utilizes advanced algorithms to generate high-quality embeddings that capture the semantic essence of the input text and images, making it ideal for applications in multi-modal data processing, information retrieval, and similar fields.

Key Features:

**High Performance**: Optimized for quick and reliable conversion of textual data and image inputs into vector embeddings.

**Scalability**: Built to handle high volumes of requests simultaneously, ensuring robust performance even under heavy loads.

**Ease of Integration**: Provides a simple and intuitive API, allowing for straightforward integration into existing systems and workflows.

**Customizable**: Supports configuration and customization to meet specific use case requirements, including different embedding models and preprocessing techniques.

Users are albe to configure and build embedding-related services according to their actual needs.



## 🚀1. Start Microservice with Docker 

### 1.2 Build Docker Image

#### Build Langchain Docker 

```bash
cd ../../
docker build -t opea/embedding-multimodal:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/embeddings/multimodal_embeddings/multimodal_langchain/docker/Dockerfile .
```

### 1.4 Run Docker with Docker Compose

```bash
cd docker
docker compose -f docker_compose_multimodal_embedding.yaml up -d
```

## 🚀2. Consume Embedding Service

### 2.1 Check Service Status

```bash
curl http://localhost:6601/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```


### 2.2 Consume Embedding Service

```bash
curl -X POST http://0.0.0.0:6601/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"text": {"text" : "This is some sample text."}, "image" : {"url": "https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true"}}'
 
curl -X POST http://0.0.0.0:6601/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"text" : "This is some sample text."}'
```
