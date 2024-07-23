# PII Detection Prediction Guard Microservice

[Prediction Guard](https://docs.predictionguard.com) Prediction Guard allows you to seamlessly integrate private, controlled, and compliant Large Language Models (LLM) functionality. In addition to providing a scalable LLM API, we enable you to prevent hallucinations, institute governance, and ensure compliance. Using Prediction Guard gives you quick and easy access to state-of-the-art LLMs. Acquire an API key by going [here](https://mailchi.mp/predictionguard/getting-started).

Detecting Personal Identifiable Information (PII) is important in ensuring that users aren't sending out private data to LLMs.

# 🚀1. Start Microservice with Python

## 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

# 🚀2. Start Microservice with Docker

## 2.1 Setup Environment Variables

Setup the following environment variables first

```bash
export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
```

## 2.1 Build Docker Images

```bash
cd ../..
docker build -t opea/pii-predictionguard:latest -f comps/guardrails/pii_detection/predictionguard/docker/Dockerfile .
```

## 2.2 Start Service

```bash
docker run -d --name="pii-predictionguard" -p 9080:9080 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/pii_predictionguard:latest
```

# 🚀3. Consume PII Detection Service

```bash
curl -X POST http://localhost:9080/v1/pii \
    -H 'Content-Type: application/json' \
    -d '{
      "prompt": "My name is John Doe and my phone number is 555-555-5555.",
      "replace": true,
      "replace_method": "random"
    }' 
```