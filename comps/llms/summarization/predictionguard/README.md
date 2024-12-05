# Prediction Guard Introduction

[Prediction Guard](https://docs.predictionguard.com) allows you to utilize hosted open access LLMs, LVMs, and embedding functionality with seamlessly integrated safeguards. In addition to providing a scalable access to open models, Prediction Guard allows you to configure factual consistency checks, toxicity filters, PII filters, and prompt injection blocking. Join the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd) and request an API key to get started.

# Getting Started

## 🚀1. Start Microservice with Docker 🐳 

### 1.1 Set up Prediction Guard API Key

You can get your API key from the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd).
```bash
export PREDICTIONGUARD_API_KEY=<your_api_key>
```

###  1.2 Build Docker Image

```bash
docker build -t opea/llm-docsum-predictionguard:latest -f comps/llms/summarization/predictionguard/Dockerfile .
```

### 1.3 Run the Predictionguard Microservice

```bash
docker run -d -p 9000:9000 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY  --name llm-docsum-predictionguard opea/llm-docsum-predictionguard:latest
```

## 🚀 2. Consume the Prediction Guard Microservice

See the [Prediction Guard docs](https://docs.predictionguard.com/) for available model options.

### Without streaming

```bash
curl -X POST http://localhost:9000/v1/chat/docsum \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hermes-2-Pro-Llama-3-8B",
        "query": "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data.",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": false
    }'
```

### With streaming

```bash
curl -N -X POST http://localhost:9000/v1/chat/docsum \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hermes-2-Pro-Llama-3-8B",
        "query": "Deep learning is a subset of machine learning that utilizes neural networks with multiple layers to analyze various levels of abstract data representations. It enables computers to identify patterns and make decisions with minimal human intervention by learning from large amounts of data.",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": true
    }'
```
