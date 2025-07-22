# Document Summary LLM Microservice

The Document Summary LLM Microservice leverages LangChain to provide advanced text summarization and Large Language Model (LLM) inference using Text Generation Inference (TGI) on Intel Xeon and Gaudi2 processors. The backend can be configured to use either [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm).

---

## Table of Contents

1. [Start Microservice](#start-microservice)
2. [Consume Microservice](#consume-microservice)
3. [Air-Gapped Environment Usage](#air-gapped-environment-usage)

---

## Start Microservice

### Set Environment Variables

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

> MAX_TOTAL_TOKENS must be greater than MAX_INPUT_TOKENS + max_new_tokens + 50 (50 tokens reserved for prompt length).

### Build Docker Images

#### Build Backend LLM Image

For vLLM, refer to [vLLM Build Instructions](../../../third_parties/vllm/).

TGI does not require additional setup.

#### Build DocSum Microservice Image

```bash
cd ../../../../
docker build -t opea/llm-docsum:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/doc-summarization/Dockerfile .
```

### Run Docker Service

You can start the service using either the CLI or Docker Compose.

#### Option A: Run with Docker CLI

1. Start the backend LLM service ([TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm)).

2. Start DocSum microservice:

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
    -e MAX_INPUT_TOKENS=$MAX_INPUT_TOKENS \
    -e MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS \
    opea/llm-docsum:latest
```

#### Option B: Run with Docker Compose

```bash
export service_name="docsum-tgi"
# Alternatives: "docsum-tgi-gaudi", "docsum-vllm", "docsum-vllm-gaudi"

cd ../../deployment/docker_compose/
docker compose -f compose_doc-summarization.yaml up ${service_name} -d
```

### Kubernetes Deployment (Optional)

```bash
kubectl apply -f deployment/k8s/docsum-deployment.yaml
kubectl apply -f deployment/k8s/docsum-service.yaml
kubectl apply -f deployment/k8s/docsum-ingress.yaml  # Optional
```

For details, see [Kubernetes Deployment Guide](../../../llms/deployment).

---

## Consume Microservice

### Check Service Status

```bash
curl http://${your_ip}:9000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### Consume LLM Service

Basic usage:

In DocSum microservice, except for basic LLM parameters, we also support several optimization parameters setting.

- "language": specify the language, can be "auto", "en", "zh", default is "auto"

If you want to deal with long context, can select suitable summary type, details in section 3.2.2.

- "summary_type": can be "auto", "stuff", "truncate", "map_reduce", "refine", default is "auto"
- "chunk_size": max token length for each chunk. Set to be different default value according to "summary_type".
- "chunk_overlap": overlap token length between each chunk, default is 0.1\*chunk_size

#### Basic usage

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

#### Long context summarization with "summary_type"

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

---

## Air-Gapped Environment Usage

The following steps are needed for running the `opea/llm-docsum` microservice in an air gapped environment (a.k.a. environment with no internet access).

1. Pre download the following models, i.e. `huggingface-cli download --cache-dir <model data directory> <model>`

- gpt2
- the same model as the LLM inference backend

2. Launch the `opea/llm-docsum` microservice with the following settings:

- mount the host `<model data directory>` as the `/data` directory within the microservice container
- leave environment as unset `HF_TOKEN` as unset

e.g. `unset HF_TOKEN; docker run -v <model data directory>:/data ... ...`
