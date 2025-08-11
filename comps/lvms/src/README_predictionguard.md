# LVM Microservice with PredictionGuard

This service utilizes [Prediction Guard](https://docs.predictionguard.com) for Visual Question and Answering (VQA). Prediction Guard provides access to hosted open models with seamlessly integrated safeguards, including factual consistency checks, toxicity filters, and more.

---

## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume LVM Service](#consume-lvm-service)

---

## Start Microservice

### Build Docker Image

First, build the generic LVM microservice Docker image:

```bash
cd ../../../
docker build -t opea/lvm:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f comps/lvms/src/Dockerfile .
```

### Run with Docker Compose

Deploy the PredictionGuard LVM service using Docker Compose.

1.  Export the required environment variables. Get your API key from [Prediction Guard](https://docs.predictionguard.com).

    ```bash
    export PREDICTIONGUARD_PORT=9399
    export PREDICTIONGUARD_API_KEY=${your_predictionguard_api_key}
    ```

2.  Navigate to the Docker Compose directory and start the service:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up predictionguard-service -d
    ```

---

## Consume LVM Service

Once the service is running, you can send requests to the API.

### Use the LVM Service API

Send a POST request with an image (base64 encoded) and a prompt.

```bash
curl http://localhost:9399/v1/lvm \
  -X POST \
  -d '{"image": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8/5+hnoEIwDiqkL4KAcT9GO0U4BxoAAAAAElFTkSuQmCC", "prompt":"What is this?"}' \
  -H 'Content-Type: application/json'
```
