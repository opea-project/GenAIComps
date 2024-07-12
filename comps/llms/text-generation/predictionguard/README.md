# Introduction

[Prediction Guard](https://docs.predictionguard.com) Prediction Guard allows you to seamlessly integrate private, controlled, and compliant Large Language Models (LLM) functionality. In addition to providing a scalable LLM API, we enable you to prevent hallucinations, institute governance, and ensure compliance. Using Prediction Guard gives you quick and easy access to state-of-the-art LLMs
# Get Started

## Usage

# Build Docker Image

```bash
cd GenAIComps/
docker build -t predictionguard-llm -f comps/llms/text-generation/predictionguard/Dockerfile .                          
```

# Run the Ollama Microservice

```bash
docker run -d -p 9000:9000 -e PREDICTIONGUARD_API_KEY="<API_KEY>" --name predictionguard-llm-container predictionguard-llm
```

# Consume the Ollama Microservice

```bash
curl -X POST http://localhost:9000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Neural-Chat-7B",
        "query": "Tell me a joke.",
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    }'
```
