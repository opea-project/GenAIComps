# FAQGen LLM Microservice

The FAQGen LLM Microservice interacts with the TGI/vLLM LLM server to generate FAQs(frequently asked questions and answers) from Input Text. The backend can be configured to use either [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm).

---

## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume Microservice](#consume-microservice)

---

## Start Microservice

### Set Environment Variables

```bash
export host_ip=${your_host_ip}
export LLM_ENDPOINT_PORT=8008
export FAQ_PORT=9000
export HF_TOKEN=${your_hf_api_token}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID=${your_hf_llm_model}
```

### Build Docker Images

#### Build Backend LLM Image

For vLLM, refer to [vLLM Build Instructions](../../../third_parties/vllm/).

TGI does not require additional setup.

#### Build FAQGen Microservice Image

```bash
cd ../../../../
docker build -t opea/llm-faqgen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/faq-generation/Dockerfile .
```

### Run Docker Service

You can start the service using either the CLI or Docker Compose.

#### Option A: Run with Docker CLI

1. Start the backend LLM service ([TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm)).

2. Start FAQGen microservice:

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

#### Option B: Run with Docker Compose

```bash
export service_name="faqgen-tgi"
# Alternatives:
# export service_name="faqgen-tgi-gaudi"
# export service_name="faqgen-vllm"
# export service_name="faqgen-vllm-gaudi"

cd ../../deployment/docker_compose/
docker compose -f compose_faq-generation.yaml up ${service_name} -d
```

---

## Consume Microservice

### Check Service Status

```bash
curl http://${host_ip}:${FAQ_PORT}/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### Consume FAQGen LLM Service

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
