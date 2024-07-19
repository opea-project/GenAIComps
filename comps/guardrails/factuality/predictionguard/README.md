# Factuality Check Prediction Guard Microservice

[Prediction Guard](https://docs.predictionguard.com) Prediction Guard allows you to seamlessly integrate private, controlled, and compliant Large Language Models (LLM) functionality. In addition to providing a scalable LLM API, we enable you to prevent hallucinations, institute governance, and ensure compliance. Using Prediction Guard gives you quick and easy access to state-of-the-art LLMs. Acquire an API key by going [here](https://mailchi.mp/predictionguard/getting-started).

Checking for factuality can help to ensure that any LLM hallucinations are being found before being returned to a user, and also ensure that your LLMs are responding with factual information.

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
docker build -t opea/factuality_predictionguard:latest -f comps/guardrails/factuality/predictionguard/docker/Dockerfile .
```

## 2.2 Start Service

```bash
docker run -d --name="factuality-predictionguard" -p 9075:9075 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/factuality_predictionguard:latest
```

# 🚀3. Consume LVM Service

```bash
curl -X POST http://localhost:9075/v1/factuality \
    -H 'Content-Type: application/json' \
    -d '{
      "reference": "The sky is blue.",
      "text": "The sky is green."
    }' 
```