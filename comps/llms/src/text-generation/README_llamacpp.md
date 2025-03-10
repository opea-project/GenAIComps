# llama.cpp Introduction

[llama.cpp](https://github.com/ggerganov/llama.cpp) provides inference in pure C/C++, and enables "LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud".

This OPEA component wraps llama.cpp server so that it can interface with other OPEA components, or for creating OPEA Megaservices.

llama.cpp supports this [hardware](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#supported-backends), and has only been tested on CPU.

To use a CUDA server please refer to [this llama.cpp reference](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#docker) and modify docker_compose_llm.yaml accordingly.

## Get Started

### 1. Download a gguf model to serve

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
export TEXTGEN_PORT=9000
export LLM_ENDPOINT_PORT=8008
export LLM_ENDPOINT="http://${host_ip}:80"
export LLM_MODEL_ID="models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
export LLAMA_ARG_CTX_SIZE=4096
```

### 3. Run the llama.cpp OPEA Microservice

```bash
export service_name="textgen-service-llamacpp"
cd comps/llms/deployment/docker_compose/
docker compose -f compose_text-generation.yaml up ${service_name} -d
```

The server output can be observed in a terminal with `docker log <container>`.

## Consume the Service

Verify the backend llama.cpp backend server:

```bash
curl http://0.0.0.0:8008/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer no-key" \
    -d '{
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is deep learning?"
        }
        ]
    }'
```

Consume the service:

This component is based on openAI API convention:

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "Write a limerick about python exceptions"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "stream": false
    }'
```
