# Document Summary LLM Microservice

This microservice leverages LangChain to implement advanced text summarization strategies and facilitate Large Language Model (LLM) inference using Text Generation Inference (TGI) on Intel Xeon and Gaudi2 processors. Users can configure the backend service to utilize either [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm).

# Quick Start Guide

## Deployment options

## 🚀1. Start Microservice with Docker 🐳

### 1.1 Setup Environment Variables

In order to start DocSum services, you need to setup the following environment variables first.

```bash
export host_ip=${your_host_ip}
export LLM_ENDPOINT_PORT=8008
export DOCSUM_PORT=9000
export HF_TOKEN=${your_hf_api_token}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID=${your_hf_llm_model}
export MAX_INPUT_TOKENS=2048
export MAX_TOTAL_TOKENS=4096
```

Please make sure MAX_TOTAL_TOKENS should be larger than (MAX_INPUT_TOKENS + max_new_tokens + 50), 50 is reserved prompt length.

### 1.2 Build Docker Image

Step 1: Prepare backend LLM docker image.

If you want to use vLLM backend, refer to [vLLM](../../../third_parties/vllm/) for building the necessary Docker image.

TGI does not require additional setup.

Step 2: Build DocSum docker image:

```bash
cd ../../../../
docker build -t opea/llm-docsum:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/doc-summarization/Dockerfile .
```

### 1.3 Run Docker Service

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 1.3.1 Run Docker with CLI (Option A)

Step 1: Start the backend LLM service
Please refer to [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm) guideline to start a backend LLM service.

Step 2: Start the DocSum microservices

```bash
export DocSum_COMPONENT_NAME="OpeaDocSumTgi" # or "OpeaDocSumvLLM"
docker run -d \
    --name="llm-docsum-server" \
    -p 9000:9000 \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e LLM_MODEL_ID=$LLM_MODEL_ID \
    -e LLM_ENDPOINT=$LLM_ENDPOINT \
    -e HF_TOKEN=$HF_TOKEN \
    -e DocSum_COMPONENT_NAME=$DocSum_COMPONENT_NAME \
    -e MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} \
    -e MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} \
    opea/llm-docsum:latest
```

### 1.3.2 Run Docker with Docker Compose (Option B)

Set `service_name` to match backend service.

```bash
export service_name="docsum-tgi"
# Alternative you can use service_name as: "docsum-tgi-gaudi", "docsum-vllm", "docsum-vllm-gaudi"

cd ../../deployment/docker_compose/
docker compose -f compose_doc-summarization.yaml up ${service_name} -d
```

## 🚀2. Start Microservice with Kubernetes

The **DocSum microservice** can be deployed on a **Kubernetes cluster** using the provided manifests.

### 2.1 Deployment Overview

- Requires **a running Kubernetes cluster** and `kubectl` configured.
- The service can be exposed using **ClusterIP, NodePort, or Ingress**.
- Backend LLM service (**TGI or vLLM**) must be running.

### 2.2 Quick Deployment Steps

Run the following commands to deploy:

```bash
kubectl apply -f deployment/k8s/docsum-deployment.yaml
kubectl apply -f deployment/k8s/docsum-service.yaml
kubectl apply -f deployment/k8s/docsum-ingress.yaml  # If using Ingress
```

For detailed deployment steps and configuration options, refer to the [Kubernetes Deployment Guide](../../../llms/deployment).

## 🚀3. Consume LLM Service

### 3.1 Checking Service Status

```bash
curl http://${your_ip}:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.2 Consume LLM Service

In DocSum microservice, except for basic LLM parameters, we also support several optimization parameters setting.

- "language": specify the language, can be "auto", "en", "zh", default is "auto"

If you want to deal with long context, can select suitable summary type, details in section 3.2.2.

- "summary_type": can be "auto", "stuff", "truncate", "map_reduce", "refine", default is "auto"
- "chunk_size": max token length for each chunk. Set to be different default value according to "summary_type".
- "chunk_overlap": overlap token length between each chunk, default is 0.1\*chunk_size

#### 3.2.1 Basic usage

```bash
# Enable stream to receive a stream response. By default, this is set to True.
curl http://${your_ip}:9000/v1/docsum \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en"}' \
  -H 'Content-Type: application/json'

# Disable stream to receive a non-stream response.
curl http://${your_ip}:9000/v1/docsum \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "stream":false}' \
  -H 'Content-Type: application/json'

# Use Chinese mode
curl http://${your_ip}:9000/v1/docsum \
  -X POST \
  -d '{"messages":"2024年9月26日，北京——今日，英特尔正式发布英特尔® 至强® 6性能核处理器（代号Granite Rapids），为AI、数据分析、科学计算等计算密集型业务提供卓越性能。", "max_tokens":32, "language":"zh", "stream":false}' \
  -H 'Content-Type: application/json'
```

#### 3.2.2 Long context summarization with "summary_type"

**summary_type=auto**

"summary_type" is set to be "auto" by default, in this mode we will check input token length, if it exceed `MAX_INPUT_TOKENS`, `summary_type` will automatically be set to `refine` mode, otherwise will be set to `stuff` mode.

With long contexts, request may get canceled due to its generation taking longer than the default `timeout` value (120s for TGI). Increase it as needed.

**summary_type=stuff**

In this mode LLM generate summary based on complete input text. In this case please carefully set `MAX_INPUT_TOKENS` and `MAX_TOTAL_TOKENS` according to your model and device memory, otherwise it may exceed LLM context limit and raise error when meet long context.

**summary_type=truncate**

Truncate mode will truncate the input text and keep only the first chunk, whose length is equal to `min(MAX_TOTAL_TOKENS - input.max_tokens - 50, MAX_INPUT_TOKENS)`

```bash
curl http://${your_ip}:9000/v1/docsum \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "summary_type": "truncate", "chunk_size": 2000}' \
  -H 'Content-Type: application/json'
```

**summary_type=map_reduce**

Map_reduce mode will split the inputs into multiple chunks, map each document to an individual summary, then consolidate those summaries into a single global summary. `stream=True` is not allowed here.

In this mode, default `chunk_size` is set to be `min(MAX_TOTAL_TOKENS - input.max_tokens - 50, MAX_INPUT_TOKENS)`

```bash
curl http://${your_ip}:9000/v1/docsum \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "summary_type": "map_reduce", "chunk_size": 2000, "stream":false, "timeout":200}' \
  -H 'Content-Type: application/json'
```

**summary_type=refine**

Refine mode will split the inputs into multiple chunks, generate summary for the first one, then combine with the second, loops over every remaining chunks to get the final summary.

In this mode, default `chunk_size` is set to be `min(MAX_TOTAL_TOKENS - 2 * input.max_tokens - 128, MAX_INPUT_TOKENS)`.

```bash
curl http://${your_ip}:9000/v1/docsum \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.", "max_tokens":32, "language":"en", "summary_type": "refine", "chunk_size": 2000, "timeout":200}' \
  -H 'Content-Type: application/json'
```

## Running in the air gapped environment

The following steps are needed for running the `opea/llm-docsum` microservice in an air gapped environment (a.k.a. environment with no internet access).

1. Pre download the following models, i.e. `huggingface-cli download --cache-dir <model data directory> <model>`

- gpt2
- the same model as the LLM inference backend

2. Launch the `opea/llm-docsum` microservice with the following settings:

- mount the host `<model data directory>` as the `/data` directory within the microservice container
- leave environment as unset `HF_TOKEN` as unset

e.g. `unset HF_TOKEN; docker run -v <model data directory>:/data ... ...`
