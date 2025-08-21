# LVM Microservice with TGI-accelerated LLaVA

This service deploys the LLaVA model accelerated by Text Generation Inference (TGI), specifically optimized for high-performance inference on Intel Gaudi HPUs.

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

Deploy the TGI LLaVA service and the LVM microservice using Docker Compose.

1.  Export the required environment variables:

    ```bash
    export ip_address=$(hostname -I | awk '{print $1}')
    export LVM_PORT=9399
    export LLAVA_TGI_PORT=11502
    export LVM_ENDPOINT="http://$ip_address:$LLAVA_TGI_PORT"
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up llava-tgi-service lvm-llava-tgi -d
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
