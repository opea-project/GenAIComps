# TGI LLM Microservice based on Llama Stack

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and more.

[Llama Stack](https://github.com/meta-llama/llama-stack) is a comprehensive set of interfaces developed by Meta for ML developers building on top of Llama foundation models. This API aims to standardize interactions with Llama models, simplifying the developer experience and fostering innovation across the Llama ecosystem. The Llama Stack encompasses various components of the model lifecycle, including inference, fine-tuning, evaluations, and synthetic data generation.

This guide provides an example on how to launch TGI endpoint based on Llama Stack on CPU and Gaudi accelerators.

## 🚀1. Start Microservice with Python (Option 1)

To start the LLM microservice, you need to install python packages first.

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start TGI Service
First we start a TGI endpoint for your LLM model.
```bash
export HF_TOKEN=${your_hf_api_token}
export LLM_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" # change to your model
docker run \
  -p 8008:80 \
  -v ./data:/data \
  --name tgi_service \
  --shm-size 1g \
  -e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$https_proxy \
  ghcr.io/huggingface/text-generation-inference:2.1.0 \
  --model-id ${LLM_MODEL_ID}
```

### 1.3 Start Llama Stack Server
Then we stat the Llama Stack server based on TGI endpoint.

```bash
## insert TGI_LLM_ENDPOINT to llama stack configuration
export TGI_LLM_ENDPOINT="http://${your_ip}:8008"
sed -i "s/ENTER_YOUR_TGI_HOSTED_ENDPOINT/${TGI_LLM_ENDPOINT}/g" ./dependency/llama_stack_run.yaml

# start llama stack server
docker run \
  --network host \
  -e HTTPS_PROXY=$https_proxy -e HTTP_PROXY=$https_proxy \
  -p 5000:5000 \
  -v ./dependency/llama_stack_run.yaml:/root/run.yaml llamastack/distribution-tgi --yaml_config /root/run.yaml
```

### 1.4 Start Microservice with Python Script
```bash
export LLAMA_STACK_ENDPOINT="http://${your_ip}:5000"
python llm.py
```
## 🚀2. Start Microservice with Docker (Option 2)

If you start an LLM microservice with docker, the `docker_compose_llm.yaml` file will automatically start TGI and Llama Stack service with docker.

### 2.1 Setup Environment Variables

In order to start services, you need to setup the following environment variables first.

```bash
export HF_TOKEN=${your_hf_api_token}
export LLM_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" # change to your model
export TGI_LLM_ENDPOINT="http://${your_ip}:8008"
export LLAMA_STACK_ENDPOINT="http://${your_ip}:5000"
sed -i "s/ENTER_YOUR_TGI_HOSTED_ENDPOINT/${TGI_LLM_ENDPOINT}/g" ./dependency/llama_stack_run.yaml
```

### 2.2 Build Docker Image

```bash
cd ../../../../
docker build -t opea/llm-tgi-llamastack:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/text-generation/tgi/llama_stack/Dockerfile .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 2.3 Run Docker with CLI (Option A)

```bash
docker run -d --name="llm-tgi-llamastack-server" -p 9000:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e HF_TOKEN=$HF_TOKEN -e LLM_MODEL_ID=$LLM_MODEL_ID -e LLAMA_STACK_ENDPOINT=$LLAMA_STACK_ENDPOINT opea/llm-tgi-llamastack:latest
```

### 2.4 Run Docker with Docker Compose (Option B)

```bash
docker compose -f docker_compose_llm.yaml up -d
```

## 🚀3. Consume LLM Service

### 3.1 Check Service Status

```bash
curl http://${your_ip}:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume the Services


Verify the TGI Service
```bash
curl http://${your_ip}:8008/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

Verify Llama Stack Service
```bash
curl http://${your_ip}:5000/inference/chat_completion \
-H "Content-Type: application/json" \
-d '{
    "model": "Llama3.1-8B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2 sentence poem about the moon"}
    ],
    "sampling_params": {"temperature": 0.7, "seed": 42, "max_tokens": 512}
}'
```

Verify the LLM microservice.

```bash
# non-streaming mode
curl http://${your_ip}:9000/v1/chat/completions \
  -X POST \
  -d '{"query":"What is Deep Learning?","max_tokens":17,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":false}' \
  -H 'Content-Type: application/json'

# streaming mode
curl http://${your_ip}:9000/v1/chat/completions \
  -X POST \
  -d '{"query":"What is Deep Learning?","max_tokens":17,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":true}' \
  -H 'Content-Type: application/json'
```