# LVM Microservice with LLaVA

This service uses the LLaVA (Large Language and Vision Assistant) model to provide general-purpose Visual Question and Answering (VQA) capabilities. It accepts an image and a text prompt to generate a relevant answer.

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

Deploy the LLaVA service and the LVM microservice using Docker Compose.

1.  Export the required environment variables:

    ```bash
    export ip_address=$(hostname -I | awk '{print $1}')
    export LVM_PORT=9399
    export LLAVA_PORT=11500
    export LVM_ENDPOINT="http://$ip_address:$LLAVA_PORT"
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up llava-service lvm-llava -d
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
