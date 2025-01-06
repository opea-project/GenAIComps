# Introduction

[llama.cpp](https://github.com/ggerganov/llama.cpp) provides inference in pure C/C++, and enables "LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware - locally and in the cloud".

This OPEA component wraps llama.cpp server so that it can interface with other OPEA components, or for creating OPEA Megaservices.

llama.cpp supports this [hardware](https://github.com/ggerganov/llama.cpp?tab=readme-ov-file#supported-backends), and has only been tested on CPU.

To use a CUDA server please refer to [this llama.cpp reference](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#docker) and modify docker_compose_llm.yaml accordingly.

## TLDR

```bash
cd GenAIComps/
docker compose -f comps/llms/text-generation/llamacpp/docker_compose_llm.yaml up
```

Please note it's instructive to run and validate each the llama.cpp server and OPEA component below.

## 1. Run the llama.cpp server

```bash
cd GenAIComps
docker compose -f comps/llms/text-generation/llamacpp/docker_compose_llm.yaml up llamacpp-server --force-recreate
```

Notes:

i) If you prefer to run above in the background without screen output use `up -d` . The `--force-recreate` clears cache.

ii) To tear down the llama.cpp server and remove the container:

`docker compose -f comps/llms/text-generation/llamacpp/langchain/docker_compose_llm.yaml llamacpp-server down`

iii) For [llama.cpp settings](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md) please specify them in the docker_compose_llm.yaml file.

#### Verify the llama.cpp Service:

```bash
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'
```

## 2. Run the llama.cpp OPEA Service

This is essentially a wrapper component of Llama.cpp server. OPEA nicely standardizes and verifies LLM inputs with LLMParamsDoc class (see llm.py).

### 2.1 Build the llama.cpp OPEA image:

```bash
cd GenAIComps/
docker compose -f comps/llms/text-generation/llamacpp/docker_compose_llm.yaml up llama-opea-llm
```

Equivalently, the above can be achieved with `build` and `run` from the Dockerfile. Build:

```bash
cd GenAIComps/
docker build --no-cache -t opea/llm-llamacpp:latest \
  --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy \
  -f comps/llms/text-generation/llamacpp/Dockerfile .
```

And run:

```bash
docker run --network host -e http_proxy=$http_proxy -e https_proxy=$https_proxy \
  opea/llm-llamacpp:latest
```

### 2.3 Consume the llama.cpp Microservice:

```bash
curl http://127.0.0.1:9000/v1/chat/completions  -X POST \
   -d '{"query":"What is Deep Learning?","max_tokens":32,"top_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":false}' \
   -H 'Content-Type: application/json'
```

### Notes

Tearing down services and removing containers:

```bash
cd GenAIComps/comps/llms/text-generation/llamacpp/
docker compose -f comps/llms/text-generation/llamacpp/docker_compose_llm.yaml down
```
