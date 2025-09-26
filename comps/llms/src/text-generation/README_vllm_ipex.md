# LLM Microservice with vLLM on Intel XPU

This service provides high-throughput, low-latency LLM serving accelerated by vLLM-IPEX, optimized for Intel® Arc™ Pro B60 Graphics.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Start Microservice](#start-microservice)
3. [Consume LLM Service](#consume-llm-service)

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

    export VIDEO_GROUP_ID=$(getent group video | awk -F: '{printf "%s\n", $3}')
    export RENDER_GROUP_ID=$(getent group render | awk -F: '{printf "%s\n", $3}')

    HF_HOME=${HF_HOME:=~/.cache/huggingface}
    export HF_HOME

    export MAX_MODEL_LEN=20000
    export LLM_MODEL_ID=Qwen/Qwen3-8B-AWQ
    export LOAD_QUANTIZATION=awq
    export VLLM_PORT=41090

    # Single-Arc GPU, select GPU index as needed
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    export TENSOR_PARALLEL_SIZE=1
    # Multi-Arc GPU, select GPU indices as needed
    # export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
    # export TENSOR_PARALLEL_SIZE=2
    ```

2.  Navigate to the Docker Compose directory and start the services:
    ```bash
    cd comps/llms/deployment/docker_compose/
    docker compose -f compose_text-generation.yaml up textgen-vllm-ipex-service -d
    ```

> **Note:** More details about supported models can be found at [supported-models](https://github.com/intel/llm-scaler/tree/main/vllm#3-supported-models).

---

## Consume LLM Service

Once the service is running, you can send requests to the API.

### Use the LLM Service API

Send a POST request with a prompt.

```bash
curl http://localhost:41090/v1/chat/completions -XPOST -H "Content-Type: application/json" -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [
      {
        "role": "user",
        "content": "What is Deep Learning?"
      }
    ],
    "max_tokens": 512
  }'
```
