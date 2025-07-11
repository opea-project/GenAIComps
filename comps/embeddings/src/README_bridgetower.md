# Multimodal Embeddings Microservice

The Multimodal Embedding Microservice is designed to efficiently convert pairs of textual string and image into vectorized embeddings, facilitating seamless integration into various machine learning and data processing workflows. This service utilizes advanced algorithms to generate high-quality embeddings that capture the joint semantic essence of the input text-and-image pairs, making it ideal for applications in multi-modal data processing, information retrieval, and similar fields.

---

## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume Embedding Service](#consume-embedding-service)

---

## Start Microservice

You can build and deploy the multimodal embedding microservice using Docker and Docker Compose.

### Build Docker Image

#### Build Bridgetower Model Service Image

- For Gaudi HPU:

```bash
cd ../../../
docker build -t opea/embedding-multimodal-bridgetower-hpu:latest --build-arg EMBEDDER_PORT=$EMBEDDER_PORT --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/bridgetower/src/Dockerfile.intel_hpu .
```

- For Xeon CPU:

```bash
cd ../../../
docker build -t opea/embedding-multimodal-bridgetower:latest --build-arg EMBEDDER_PORT=$EMBEDDER_PORT --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/third_parties/bridgetower/src/Dockerfile .
```

#### Build Embedding Microservice Docker

```bash
cd ../../../
docker build -t opea/embedding:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/embeddings/src/Dockerfile .
```

### Run Docker with Docker Compose

```bash
export your_mmei_port=8080
export EMBEDDER_PORT=$your_mmei_port
export MMEI_EMBEDDING_ENDPOINT="http://$ip_address:$your_mmei_port"
export your_embedding_port_microservice=6600
export MM_EMBEDDING_PORT_MICROSERVICE=$your_embedding_port_microservice
cd comps/embeddings/deployment/docker_compose/
```

- For Gaudi HPU:

```bash
docker compose up multimodal-bridgetower-embedding-gaudi-serving multimodal-bridgetower-embedding-gaudi-server -d
```

- For Xeon CPU:

```bash
docker compose up multimodal-bridgetower-embedding-serving multimodal-bridgetower-embedding-server -d
```

---

## Consume Embedding Service

Once the service is running, you can start using the API to generate embeddings for text and image pairs.

### Check Service Status

Verify that the embedding service is running properly by checking its health status with this command:

```bash
curl http://localhost:6000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### Use the Embedding Service API

You can now make API requests to generate embeddings. The service supports both single text embeddings and joint text-image embeddings.

**Compute a Joint Embedding of an Image-Text Pair**
To generate a joint embedding of a text and image input, use the following API request:

```bash
curl -X POST http://0.0.0.0:6600/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"text": {"text" : "This is some sample text."}, "image" : {"url": "https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true"}}'
```

In this example, the input is a text and an image URL. The service will return a vectorized embedding that represents both the text and image.

**Compute Text-Only Embedding**

To generate an embedding for just a text input, use this request:

```bash
curl -X POST http://0.0.0.0:6600/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"text" : "This is some sample text."}'
```

This request will return an embedding representing the semantic meaning of the input text.
