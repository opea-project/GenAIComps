# LVM Microservice with vLLM on Gaudi

This service provides high-throughput, low-latency LVM serving accelerated by vLLM, optimized for Intel Gaudi HPUs.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Start Microservice](#start-microservice)
3. [Consume LVM Service](#consume-lvm-service)

---

## Prerequisites

### Build vLLM Gaudi Docker Image

You must build the custom `vllm-gaudi` Docker image locally first.

```bash
git clone https://github.com/HabanaAI/vllm-fork.git
cd ./vllm-fork/
# Note: The commit hash is for a specific version. Check for updates.
git checkout f78aeb9da0712561163eddd353e3b6097cd69bac
docker build -f Dockerfile.hpu -t opea/vllm-gaudi:latest --shm-size=128g . --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy
cd ..
rm -rf vllm-fork
```

## Start Microservice

### Build LVM Docker Image

Build the generic LVM microservice Docker image:

```bash
cd ../../../
docker build -t opea/lvm:latest \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -f comps/lvms/src/Dockerfile .
```

### Run with Docker Compose

Deploy the vLLM service and the LVM microservice using Docker Compose.

1.  Export the required environment variables:

    ```bash
    export ip_address=$(hostname -I | awk '{print $1}')
    export LVM_PORT=9399
    export VLLM_PORT=11507
    export LVM_ENDPOINT="http://$ip_address:$VLLM_PORT"

    # Option 1: for LLaVA model
    export LLM_MODEL_ID=llava-hf/llava-1.5-7b-hf
    export CHAT_TEMPLATE=examples/template_llava.jinja

    # Option 2: for UI-TARS model
    # export LLM_MODEL_ID=bytedance-research/UI-TARS-7B-DPO
    # export TP_SIZE=1    # change to 4 or 8 if using UI-TARS-72B-DPO
    # export CHAT_TEMPLATE=None

    # Skip warmup for faster server start on Gaudi (may increase initial inference time)
    export VLLM_SKIP_WARMUP=true
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up vllm-gaudi-service lvm-vllm-gaudi -d
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
