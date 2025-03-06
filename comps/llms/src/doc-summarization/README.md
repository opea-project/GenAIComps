#  LLM Document Summary Microservice

This microservice leverages LangChain to implement summarization strategies and facilitate LLM inference using Text Generation Inference on Intel Xeon and Gaudi2 processors. You can set backend service either [TGI](../../../third_parties/tgi) or [vLLM](../../../third_parties/vllm).

## Support integrations

In this microservices, we have supported following backend LLM service as integrations, we will include TGI/vLLM/Ollama in this readme, for others, please refer to corresponding readmes.

- TGI
- VLLM


## Clone OPEA GenAIComps
Clone this repository at your desired location and set an environment variable for easy setup and usage throughout the instructions.

```bash
git clone https://github.com/opea-project/GenAIComps.git

export OPEA_GENAICOMPS_ROOT=$(pwd)/GenAIComps
```


## Prerequisites
For TGI/vLLM, You must create a user account with HuggingFace and obtain permission to use the gated LLM models by adhering to the guidelines provided on the respective model's webpage. The environment variables LLM_MODEL would be the HuggingFace model id and the HF_TOKEN is your HuggugFace account's "User Access Token".

## 🚀 Start Microservice with Docker 🐳

### 1. Build Docker Image

#### 1.1 Prepare backend LLM docker image.
If you want to use vLLM backend, refer to [vLLM](../../../third_parties/vllm/) to build vLLM docker images first.

No need for TGI.

#### 1.2 Prepare DocSum docker image.

```bash
# Build the microservice docker
cd ${OPEA_GENAICOMPS_ROOT}/

docker build \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/llm-docsum:latest \
  -f comps/llms/src/doc-summarization/Dockerfile .
```

### 2. Start LLM Service with the built image

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

If you start an LLM microservice with docker compose, the `compose_doc-summarization.yaml` file will automatically start both endpoint and the microservice docker.


#### 2.1 Setup Environment Variables
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

#### 2.2 Run Docker with CLI (Option A)

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

#### 2.2 Run Docker with Docker Compose (Option B)

Set `service_name` to match backend service.

```bash
export service_name="docsum-tgi"
# export service_name="docsum-tgi-gaudi"
# export service_name="docsum-vllm"
# export service_name="docsum-vllm-gaudi"

cd ../../deployment/docker_compose/
docker compose -f compose_doc-summarization.yaml up ${service_name} -d
```

## 🚀 Consume LLM Service

### 3. Check Service Status

```bash
curl http://${your_ip}:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 4. Consume LLM Service

In DocSum microservice, except for basic LLM parameters, we also support several optimization parameters setting.

- "language": specify the language, can be "auto", "en", "zh", default is "auto"

If you want to deal with long context, can select suitable summary type, details in section 3.2.2.

- "summary_type": can be "auto", "stuff", "truncate", "map_reduce", "refine", default is "auto"
- "chunk_size": max token length for each chunk. Set to be different default value according to "summary_type".
- "chunk_overlap": overlap token length between each chunk, default is 0.1\*chunk_size

#### 4.1 Basic usage

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

#### 4.2 Long context summarization with "summary_type"

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
