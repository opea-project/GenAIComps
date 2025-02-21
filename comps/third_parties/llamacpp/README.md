# Introduction

[llama.cpp](https://github.com/ggerganov/llama.cpp) provides inference in pure C/C++, and enables "LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud".

This OPEA component wraps llama.cpp server so that it can interface with other OPEA components, or for creating OPEA Megaservices.

llama.cpp supports this [hardware](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#supported-backends), and has only been tested on CPU.

To use a CUDA server please refer to [this llama.cpp reference](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#docker) and modify docker_compose_llm.yaml accordingly.

## Get Started

### 1. Download a gguf Model

To download an example .gguf model to a model path:

```bash
export MODEL_PATH=~/models
mkdir -p $MODEL_PATH # -p means make only if doesn't exist
cd $MODEL_PATH

wget --no-clobber https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 2. Set Environment Variables

```bash
export MODEL_PATH=~/models
export host_ip=$(hostname -I | awk '{print $1}')
export LLM_ENDPOINT_PORT=8008
export LLM_MODEL_ID="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
export LLAMA_ARG_CTX_SIZE=4096
```

### 3. Run the llama.cpp Backend Microservice

```bash
cd deployment/docker_compose
docker compose -f compose.yaml up llamacpp-server -d
```

To use this in an OPEA text generation component please see [llama.cpp text-generation](../../llms/src/text-generation/README_llamacpp.md)

Note: can use docker logs <container> to observe server.

## Consume the service

Llama cpp supports openai style API:

```bash
curl http://${host_ip}:8008/v1/chat/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "What is Deep Learning?"}]}'
```
