# Embedding Generation Prediction Guard Microservice

[Prediction Guard](https://docs.predictionguard.com) Prediction Guard allows you to seamlessly integrate private, controlled, and compliant Large Language Models (LLM) functionality. In addition to providing a scalable LLM API, we enable you to prevent hallucinations, institute governance, and ensure compliance. Using Prediction Guard gives you quick and easy access to state-of-the-art LLMs. Acquire an API key by going [here](https://mailchi.mp/predictionguard/getting-started).

The Embedding Microservice is designed to efficiently convert textual strings into vectorized embeddings, facilitating seamless integration into various machine learning and data processing workflows. This service utilizes advanced algorithms to generate high-quality embeddings that capture the semantic essence of the input text, making it ideal for applications in natural language processing, information retrieval, and similar fields.

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
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy -t opea/embedding-predictionguard:latest -f comps/guardrails/embeddings/predictionguard/docker/Dockerfile .
```

## 2.2 Start Service

```bash
docker run -d --name="embedding-predictionguard" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p 6000:6000 --ipc=host -e PREDICTIONGUARD_API_KEY=$PREDICTIONGUARD_API_KEY opea/embedding_predictionguard:latest
```

# 🚀3. Consume Embeddings Service

```
curl localhost:6000/v1/embeddings \
     -X POST \
     -d '{"text":"Hello, world!"}' \
     -H 'Content-Type: application/json'
```