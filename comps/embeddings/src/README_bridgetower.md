# Multimodal Embeddings Microservice

The Multimodal Embedding Microservice is designed to efficiently convert pairs of textual string and image into vectorized embeddings, facilitating seamless integration into various machine learning and data processing workflows. This service utilizes advanced algorithms to generate high-quality embeddings that capture the joint semantic essence of the input text-and-image pairs, making it ideal for applications in multi-modal data processing, information retrieval, and similar fields.

Key Features:

**High Performance**: Optimized for quick and reliable conversion of textual data and image inputs into vector embeddings.

**Scalability**: Built to handle high volumes of requests simultaneously, ensuring robust performance even under heavy loads.

**Ease of Integration**: Provides a simple and intuitive API, allowing for straightforward integration into existing systems and workflows.

**Customizable**: Supports configuration and customization to meet specific use case requirements, including different embedding models and preprocessing techniques.

Users are albe to configure and build embedding-related services according to their actual needs.

## 📦 1. Start Microservice

### 🔹 1.1 Build Docker Image

#### Build bridgetower multimodal embedding service

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

### 🔹 1.2 Run Docker with Docker Compose

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

## 📦 2. Consume Embedding Service

Once the service is running, you can start using the API to generate embeddings for text and image pairs.

### 🔹 2.1 Check Service Status

Verify that the embedding service is running properly by checking its health status with this command:

```bash
curl http://localhost:6000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### 🔹 2.2 Use the Embedding Service API

You can now make API requests to generate embeddings. The service supports both single text embeddings and joint text-image embeddings.

**Compute a Joint Embedding of an Image-Text Pair**
To compute an embedding for a text and image pair, use the following API request:

```bash
curl -X POST http://0.0.0.0:6600/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"text": {"text" : "This is some sample text."}, "image" : {"url": "https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true"}}'
```

In this example, the input is a text and an image URL. The service will return a vectorized embedding that represents both the text and image.

**Compute an embedding of a text**

To generate an embedding for just a text input, use this request:

```bash
curl -X POST http://0.0.0.0:6600/v1/embeddings \
     -H "Content-Type: application/json" \
     -d '{"text" : "This is some sample text."}'
```

This request will return an embedding representing the semantic meaning of the input text.
