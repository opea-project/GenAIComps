# Prediction Guard Introduction

[Prediction Guard](https://docs.predictionguard.com) allows you to utilize hosted open access LLMs, LVMs, and embedding functionality with seamlessly integrated safeguards. In addition to providing a scalable access to open models, Prediction Guard allows you to configure factual consistency checks, toxicity filters, PII filters, and prompt injection blocking. Join the [Prediction Guard Discord channel](https://discord.gg/TFHgnhAFKd) and request an API key to get started.

## Get Started

### Run the Predictionguard Microservice

```bash
export service_name="textgen-predictionguard"

cd comps/llms/deployment/docker_compose/
docker compose -f compose_text-generation.yaml up ${service_name} -d
```

## Consume the Prediction Guard Microservice

See the [Prediction Guard docs](https://docs.predictionguard.com/) for available model options.

### Without stream

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hermes-3-Llama-3.1-8B",
        "messages": "Tell me a joke.",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": false
    }'
```

### With stream

```bash
curl -N -X POST http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Hermes-3-Llama-3.1-8B",
        "messages": "Tell me a joke.",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": true
    }'
```
