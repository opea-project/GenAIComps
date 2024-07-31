# Embedding Generation Prediction Guard Microservice

[Prediction Guard](https://docs.predictionguard.com) offers a comprehensive platform that enables secure, scalable access to a wide range of open-source LLMs (Large Language Models), LVMs (Large Vision Models), and advanced embedding functionality. With built-in safeguards like factual consistency checks, toxicity filters, PII filters, and prompt injection blocking, Prediction Guard ensures that your AI-driven applications are both powerful and responsible. Join the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd) and request an API key to get started.

Embeddings are a critical component in many AI applications, including Retrieval-Augmented Generation (RAG) and semantic search. They transform textual data into high-dimensional vectors, allowing for efficient comparison and retrieval of relevant information. The embedding microservice provided by Prediction Guard is specifically designed to convert text into vectorized embeddings using the BridgeTower model. This service seamlessly integrates with the broader Prediction Guard ecosystem, enabling you to leverage these embeddings for a variety of downstream tasks, such as enhancing search capabilities or improving the contextual relevance of generated content.

This embedding microservice is designed to efficiently convert text into vectorized embeddings using the [BridgeTower model](https://huggingface.co/BridgeTower/bridgetower-large-itm-mlm-itc). Thus, it is ideal for both RAG or semantic search applications.

**Note** - The BridgeTower model implemented in Prediction Guard is capable of embedding text, images, or a combination of both. While this service currently focuses on text embeddings, future updates will extend its functionality to support multimodal inputs.

# 🚀 Start Microservice with Docker

## Setup Environment Variables

Setup the following environment variables first

```bash
export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
```

## Build Docker Images

```bash
cd ../../..
docker build -t opea/embedding-predictionguard:latest -f comps/embeddings/predictionguard/docker/Dockerfile .
```

## Start Service

```bash
docker run -d --name="embedding-predictionguard" -p 6000:6000 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/embedding-predictionguard:latest
```

# 🚀 Consume Embeddings Service

```bash
curl localhost:6000/v1/embeddings \
     -X POST \
     -d '{"text":"Hello, world!"}' \
     -H 'Content-Type: application/json'
```
