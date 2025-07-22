# Embedding Microservice with Prediction Guard

[Prediction Guard](https://docs.predictionguard.com) allows you to utilize hosted open access LLMs, LVMs, and embedding functionality with seamlessly integrated safeguards. In addition to providing a scalable access to open models, Prediction Guard allows you to configure factual consistency checks, toxicity filters, PII filters, and prompt injection blocking. Join the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd) and request an API key to get started.

This embedding microservice is designed to efficiently convert text into vectorized embeddings using the [BridgeTower model](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc), making it ideal for both RAG and semantic search applications.

**Note** - The BridgeTower model implemented in Prediction Guard can actually embed text, images, or text + images (jointly). For now this service only embeds text, but a follow on contribution will enable the multimodal functionality.

---

## Table of Contents

1. [Start Microservice with `docker run`](#start-microservice-with-docker-run)
2. [Start Microservice with Docker Compose](#start-microservice-with-docker-compose)
3. [Consume Embedding Service](#consume-embedding-service)
4. [Tips for Better Understanding](#tips-for-better-understanding)

---

## Start Microservice with `docker run`

### Start Embedding Service with TEI

Before starting the service, ensure the following environment variable is set:

```bash
export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
```

### Build Docker Image

To build the Docker image for the embedding service, run the following command:

```bash
cd ../../../
docker build -t opea/embedding:latest -f comps/embeddings/src/Dockerfile .
```

### Start Service

Run the Docker container in detached mode with the following command:

```bash
docker run -d --name="embedding-predictionguard" -p 6000:6000 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/embedding:latest
```

---

## Start Microservice with Docker Compose

Deploy the Prediction Guard embedding service using Docker Compose for easier management of multi-container setups.

1. Export environment variables:

   ```bash
   export PG_EMBEDDING_MODEL_NAME="bridgetower-large-itm-mlm-itc"
   export EMBEDDER_PORT=6000
   export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
   ```

2. Navigate to the Docker Compose directory:

   ```bash
   cd comps/embeddings/deployment/docker_compose/
   ```

3. Start the services:

   ```bash
   docker compose up pg-embedding-server -d
   ```

---

## Consume Embedding Service

### Check Service Status

Verify the embedding service is running:

```bash
curl http://localhost:6000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### Use the Embedding Service API

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

---

## Tips for Better Understanding

1. **Prediction Guard Features**  
   Prediction Guard comes with built-in safeguards such as factual consistency checks, toxicity filters, PII detection, and prompt injection protection, ensuring safe use of the service.

2. **Multimodal Support**  
   While the service currently only supports text embeddings, we plan to extend this functionality to support images and joint text-image embeddings in future releases.

3. **Scalability**  
   The microservice can easily scale to handle large volumes of requests for embedding generation, making it suitable for large-scale semantic search and RAG applications.
