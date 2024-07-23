# Toxicity Checking Prediction Guard Microservice

[Prediction Guard](https://docs.predictionguard.com) Prediction Guard allows you to seamlessly integrate private, controlled, and compliant Large Language Models (LLM) functionality. In addition to providing a scalable LLM API, we enable you to prevent hallucinations, institute governance, and ensure compliance. Using Prediction Guard gives you quick and easy access to state-of-the-art LLMs. Acquire an API key by going [here](https://mailchi.mp/predictionguard/getting-started).

Checking text for toxicity allows you to ensure that your preventing toxic prompts from being sent to your LLM and toxic LLM outputs.

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
docker build -t opea/toxicity-predictionguard:latest -f comps/guardrails/toxicity_harm/predictionguard/docker/Dockerfile .
```

## 2.2 Start Service

```bash
docker run -d --name="toxicity-predictionguard" -p 9090:9090 -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/toxicity_predictionguard:latest
```

# 🚀3. Consume Toxicity Check Service

```bash
curl -X POST http://localhost:9090/v1/toxicity \
    -H 'Content-Type: application/json' \
    -d '{
      "text": "I hate you!!"
    }' 
```