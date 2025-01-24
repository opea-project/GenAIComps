# FAQGen LLM Microservice

This microservice interacts with the TGI/vLLM LLM server to generate FAQs(frequently asked questions and answers) from Input Text. You can set backend service either [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm).

## 🚀1. Start Microservice with Docker

### 1.1 Setup Environment Variables

In order to start FaqGen microservices, you need to setup the following environment variables first.

```bash
export host_ip=${your_host_ip}
export LLM_ENDPOINT_PORT=8008
export FAQ_PORT=9000
export HF_TOKEN=${your_hf_api_token}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID=${your_hf_llm_model}
```

### 1.2 Build Docker Image

Step 1: Prepare backend LLM docker image.

If you want to use vLLM backend, refer to [vLLM](../../../third_parties/vllm) to build vLLM docker images first.

No need for TGI.

Step 2: Build FaqGen docker image.

```bash
cd ../../../../
docker build -t opea/llm-faqgen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/faq-generation/Dockerfile .
```

### 1.3 Run Docker

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

#### 1.3.1 Run Docker with CLI (Option A)

Step 1: Start the backend LLM service
Please refer to [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm) guideline to start a backend LLM service.

Step 2: Start the FaqGen microservices

```bash
export FAQGen_COMPONENT_NAME="OpeaFaqGenTgi" # or "OpeaFaqGenvLLM"
docker run -d \
    --name="llm-faqgen-server" \
    -p 9000:9000 \
    --ipc=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e LLM_MODEL_ID=$LLM_MODEL_ID \
    -e LLM_ENDPOINT=$LLM_ENDPOINT \
    -e HF_TOKEN=$HF_TOKEN \
    -e FAQGen_COMPONENT_NAME=$FAQGen_COMPONENT_NAME \
    opea/llm-faqgen:latest
```

#### 1.3.2 Run Docker with Docker Compose (Option B)

Set `service_name` to match backend service.

```bash
export service_name="faqgen-tgi"
# export service_name="faqgen-tgi-gaudi"
# export service_name="faqgen-vllm"
# export service_name="faqgen-vllm-gaudi"

cd ../../deployment/docker_compose/
docker compose -f compose_faq-generation.yaml up ${service_name} -d
```

## 🚀2. Consume LLM Service

### 2.1 Check Service Status

```bash
curl http://${host_ip}:${FAQ_PORT}/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 2.2 Consume FAQGen LLM Service

```bash
# Streaming Response
# Set stream to True. Default will be True.
curl http://${host_ip}:${FAQ_PORT}/v1/faqgen \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.","max_tokens": 128}' \
  -H 'Content-Type: application/json'

# Non-Streaming Response
# Set stream to False.
curl http://${host_ip}:${FAQ_PORT}/v1/faqgen \
  -X POST \
  -d '{"messages":"Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5.","max_tokens": 128, "stream":false}' \
  -H 'Content-Type: application/json'
```
