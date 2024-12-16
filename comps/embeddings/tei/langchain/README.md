# Embeddings Microservice with Langchain TEI

## 🚀1. Start Microservice with Python (Option 1)

Currently, we provide two ways to implement the embedding service:

1. Build the embedding model **_locally_** from the server, which is faster, but takes up memory on the local server.

2. Build it based on the **_TEI endpoint_**, which provides more flexibility, but may bring some network latency.

For both of the implementations, you need to install requirements first.

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start Embedding Service

You can select one of following ways to start the embedding service:

#### Start Embedding Service with TEI

First, you need to start a TEI service.

```bash
your_port=8090
model="BAAI/bge-large-en-v1.5"
docker run -p $your_port:80 -v ./data:/data --name tei_server -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
```

Then you need to test your TEI service using the following commands:

```bash
curl localhost:$your_port/v1/embeddings \
    -X POST \
    -d '{"input":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

Start the embedding service with the TEI_EMBEDDING_ENDPOINT.

```bash
export TEI_EMBEDDING_ENDPOINT="http://localhost:$yourport/v1/embeddings"
export TEI_EMBEDDING_MODEL_NAME="BAAI/bge-large-en-v1.5"
python embedding_tei.py
```

## 🚀2. Start Microservice with Docker (Optional 2)

### 2.1 Start Embedding Service with TEI

First, you need to start a TEI service.

```bash
your_port=8090
model="BAAI/bge-large-en-v1.5"
docker run -p $your_port:80 -v ./data:/data --name tei_server -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
```

Then you need to test your TEI service using the following commands:

```bash
curl localhost:$your_port/embed/v1/embeddings \
    -X POST \
    -d '{"input":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

Export the `TEI_EMBEDDING_ENDPOINT` for later usage:

```bash
export TEI_EMBEDDING_ENDPOINT="http://localhost:$yourport/v1/embeddings"
export TEI_EMBEDDING_MODEL_NAME="BAAI/bge-large-en-v1.5"
```

### 2.2 Build Docker Image

```bash
cd ../../../../
docker build -t opea/embedding-tei:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/embeddings/tei/langchain/Dockerfile .
```

### 2.3 Run Docker with CLI

```bash
docker run -d --name="embedding-tei-server" -p 6000:6000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e TEI_EMBEDDING_MODEL_NAME=$TEI_EMBEDDING_MODEL_NAME opea/embedding-tei:latest
```

### 2.4 Run Docker with Docker Compose

```bash
cd docker
docker compose -f docker_compose_embedding.yaml up -d
```

## 🚀3. Consume Embedding Service

### 3.1 Check Service Status

```bash
curl http://localhost:6000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume Embedding Service

The input/output follows [OpenAI API Embeddings](https://platform.openai.com/docs/api-reference/embeddings) format.

```bash
## Input single text
curl http://localhost:6000/v1/embeddings\
  -X POST \
  -d '{"input":"Hello, world!"}' \
  -H 'Content-Type: application/json'

## Input multiple texts with parameters
curl http://localhost:6000/v1/embeddings\
  -X POST \
  -d '{"input":["Hello, world!","How are you?"], "encoding_format":"base64"}' \
  -H 'Content-Type: application/json'
```
