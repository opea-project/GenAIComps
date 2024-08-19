# Document Summary TGI Microservice

In this microservice, we utilize LangChain to implement summarization strategies and facilitate LLM inference using Text Generation Inference on Intel Xeon and Gaudi2 processors.
[Text Generation Inference](https://github.com/huggingface/text-generation-inference) (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and more.

## 🚀1. Start Microservice with Python (Option 1)

To start the LLM microservice, you need to install python packages first.

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start LLM Service

```bash
export HF_TOKEN=${your_hf_api_token}
docker run -p 8008:80 -v ./data:/data --name llm-docsum-tgi --shm-size 1g ghcr.io/huggingface/text-generation-inference:2.1.0 --model-id ${your_hf_llm_model}
```

### 1.3 Verify the TGI Service

```bash
curl http://${your_ip}:8008/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":17, "do_sample": true}}' \
  -H 'Content-Type: application/json'
```

### 1.4 Start LLM Service with Python Script

```bash
export TGI_LLM_ENDPOINT="http://${your_ip}:8008"
python llm.py
```

## 🚀2. Start Microservice with Docker (Option 2)

If you start an LLM microservice with docker, the `docker_compose_llm.yaml` file will automatically start a TGI/vLLM service with docker.

### 2.1 Setup Environment Variables

In order to start TGI and LLM services, you need to setup the following environment variables first.

```bash
export HF_TOKEN=${your_hf_api_token}
export TGI_LLM_ENDPOINT="http://${your_ip}:8008"
export LLM_MODEL_ID=${your_hf_llm_model}
```

### 2.2 Build Docker Image

```bash
cd ../../
docker build -t opea/llm-docsum-tgi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/summarization/tgi/Dockerfile .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 2.3 Run Docker with CLI (Option A)

```bash
docker run -d --name="llm-docsum-tgi-server" -p 9000:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TGI_LLM_ENDPOINT=$TGI_LLM_ENDPOINT -e HF_TOKEN=$HF_TOKEN opea/llm-docsum-tgi:latest
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

### 3.2 Consume LLM Service

```bash
curl http://${your_ip}:9000/v1/chat/docsum \
  -X POST \
  -d '{"query":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5."}' \
  -H 'Content-Type: application/json'
```
