# Prediction Guard Introduction

[Prediction Guard](https://docs.predictionguard.com) allows you to utilize hosted open access LLMs, LVMs, and embedding functionality with seamlessly integrated safeguards. In addition to providing a scalable access to open models, Prediction Guard allows you to configure factual consistency checks, toxicity filters, PII filters, and prompt injection blocking. Join the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd) and request an API key to get started.

## Get Started

### Build Docker Image

```bash
cd ../../..
docker build -t opea/reranks-predictionguard:latest -f comps/reranks/predictionguard/Dockerfile .
```

### Run the Predictionguard Microservice

```bash
docker run -d -p 9000:9000 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY  --name reranks-predictionguard opea/reranks-predictionguard:latest
```

## Consume the Prediction Guard Microservice

See the [Prediction Guard docs](https://docs.predictionguard.com/) for available model options.


```bash
curl -N -X POST http://localhost:9000/v1/reranking \
    -H "Content-Type: application/json" \
    -d '{
        "initial_query": "What is Deep Learning?"
        "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]
        }'
```

