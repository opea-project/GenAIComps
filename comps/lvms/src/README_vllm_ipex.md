# LVM Microservice with vLLM on Intel XPU

This service provides high-throughput, low-latency LVM serving accelerated by vLLM-IPEX, optimized for Intel® Arc™ Pro B60 Graphics.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Start Microservice](#start-microservice)
3. [Consume LVM Service](#consume-lvm-service)

---

## Prerequisites

### Download vLLM-IPEX Docker Image

You must download the official docker image from [Docker Hub](https://hub.docker.com/r/intel/llm-scaler-vllm) first.

```bash
docker pull intel/llm-scaler-vllm:1.0
```

## Start Microservice

### Run with Docker Compose

Deploy the vLLM-IPEX model serving using Docker Compose.

1.  Export the required environment variables:

    ```bash
    # Use image: intel/llm-scaler-vllm:1.0
    export REGISTRY=intel
    export TAG=1.0

    export ip_address=$(hostname -I | awk '{print $1}')
    export VIDEO_GROUP_ID=$(getent group video | awk -F: '{printf "%s\n", $3}')
    export RENDER_GROUP_ID=$(getent group render | awk -F: '{printf "%s\n", $3}')

    HF_HOME=${HF_HOME:=~/.cache/huggingface}
    export HF_HOME

    export MAX_MODEL_LEN=20000
    export LLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
    export LOAD_QUANTIZATION=fp8
    export VLLM_PORT=41091
    export LVM_ENDPOINT="http://$ip_address:$VLLM_PORT"

    # Single-Arc GPU, select GPU index as needed
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    export TENSOR_PARALLEL_SIZE=1
    # Multi-Arc GPU, select GPU indices as needed
    # export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
    # export TENSOR_PARALLEL_SIZE=2
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/lvms/deployment/docker_compose/
    docker compose up lvm-vllm-ipex-service -d
    ```

> **Note:** More details about supported models can be found at [supported-models](https://github.com/intel/llm-scaler/tree/main/vllm#3-supported-models).

---

## Consume LVM Service

Once the service is running, you can send requests to the API.

### Use the LVM Service API

Send a POST request with an image url and a prompt.

```bash
curl http://localhost:41091/v1/chat/completions -XPOST -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```
