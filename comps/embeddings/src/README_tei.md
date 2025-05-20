# 🌟 Embedding Microservice with TEI

This guide walks you through starting, deploying, and consuming the **TEI-based Embeddings Microservice**. 🚀

---

## 📦 1. Start Microservice with `docker run`

### 🔹 1.1 Start Embedding Service with TEI

1. **Start the TEI service**:
   Replace `your_port` and `model` with desired values to start the service.

   ```bash
   your_port=8090
   model="BAAI/bge-large-en-v1.5"
   docker run -p $your_port:80 -v ./data:/data --name tei-embedding-serving \
   -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always \
   ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 --model-id $model
   ```

2. **Test the TEI service**:
   Run the following command to check if the service is up and running.

   ```bash
   curl localhost:$your_port/v1/embeddings \
   -X POST \
   -d '{"input":"What is Deep Learning?"}' \
   -H 'Content-Type: application/json'
   ```

### 🔹 1.2 Build Docker Image and Run Docker with CLI

1. Build the Docker image for the embedding microservice:

   ```bash
   cd ../../../
   docker build -t opea/embedding:latest \
   --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy \
   -f comps/embeddings/src/Dockerfile .
   ```

2. Run the embedding microservice and connect it to the TEI service:

   ```bash
   docker run -d --name="embedding-tei-server" \
   -p 6000:6000 \
   -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
   --ipc=host \
   -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT \
   -e EMBEDDING_COMPONENT_NAME="OPEA_TEI_EMBEDDING" \
   opea/embedding:latest
   ```

## 📦 2. Start Microservice with docker compose

Deploy both the TEI Embedding Service and the Embedding Microservice using Docker Compose.

🔹 Steps:

1. Set environment variables:

   ```bash
   export host_ip=${your_ip_address}
   export EMBEDDING_MODEL_ID="BAAI/bge-base-en-v1.5"
   export TEI_EMBEDDER_PORT=8090
   export EMBEDDER_PORT=6000
   export TEI_EMBEDDING_ENDPOINT="http://${host_ip}:${TEI_EMBEDDER_PORT}"
   ```

2. Navigate to the Docker Compose directory:

   ```bash
   cd comps/embeddings/deployment/docker_compose/
   ```

3. Start the services:

   ```bash
   docker compose up tei-embedding-serving tei-embedding-server -d
   ```

## 📦 3. Consume Embedding Service

### 🔹 3.1 Check Service Status

Verify the embedding service is running:

```bash
curl http://localhost:6000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### 🔹 3.2 Use the Embedding Service API

The API is compatible with the [OpenAI API](https://platform.openai.com/docs/api-reference/embeddings).

1. Single Text Input

   ```bash
   curl http://localhost:6000/v1/embeddings \
   -X POST \
   -d '{"input":"Hello, world!"}' \
   -H 'Content-Type: application/json'
   ```

2. Multiple Text Inputs with Parameters

   ```bash
   curl http://localhost:6000/v1/embeddings \
   -X POST \
   -d '{"input":["Hello, world!","How are you?"], "dimensions":100}' \
   -H 'Content-Type: application/json'
   ```

## ✨ Tips for Better Understanding:

1. Port Mapping:
   Ensure the ports are correctly mapped to avoid conflicts with other services.

2. Model Selection:
   Choose a model appropriate for your use case, like "BAAI/bge-large-en-v1.5" or "BAAI/bge-base-en-v1.5".

3. Environment Variables:
   Use http_proxy and https_proxy for proxy setup if necessary.

4. Data Volume:
   The `-v ./data:/data` flag ensures the data directory is correctly mounted.
