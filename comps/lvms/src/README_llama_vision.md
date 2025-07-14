# LVM Microservice with LLaMA-Vision

This service uses the LLaMA-Vision model to provide Visual Question and Answering (VQA) capabilities.

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

Deploy the LLaMA-Vision service and the LVM microservice using Docker Compose.

1.  Export the required environment variables:

    ```bash
    export ip_address=$(hostname -I | awk '{print $1}')
    export LVM_PORT=9399
    export LLAMA_VISION_PORT=11510
    export LVM_ENDPOINT="http://$ip_address:$LLAMA_VISION_PORT"
    export LLM_MODEL_ID="meta-llama/Llama-3.2-11B-Vision-Instruct"
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up lvm-llama-vision llama-vision-service -d
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
