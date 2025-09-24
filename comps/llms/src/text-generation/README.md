# LLM Text Generation Microservice

This microservice, designed for Language Model Inference (LLM), processes input consisting of a query string and associated reranked documents. It constructs a prompt based on the query and documents, which is then used to perform inference with a large language model. The service delivers the inference results as output.

A prerequisite for using this microservice is that users must have a LLM text generation service (etc., TGI, vLLM) already running. Users need to set the LLM service's endpoint into an environment variable. The microservice utilizes this endpoint to create an LLM object, enabling it to communicate with the LLM service for executing language model operations.

Overall, this microservice offers a streamlined way to integrate large language model inference into applications, requiring minimal setup from the user beyond initiating a TGI/vLLM service and configuring the necessary environment variables. This allows for the seamless processing of queries and documents to generate intelligent, context-aware responses.

---

## Table of Contents

1. [Validated LLM Models](#validated-llm-models)
2. [Start Microservice](#start-microservice)
3. [Consume Microservice](#consume-microservice)

---

## Validated LLM Models

| Model                                           | TGI-Gaudi | vLLM-CPU | vLLM-Gaudi | vLLM-IPEX-XPU | OVMS     | Optimum-Habana | SGLANG-CPU |
| ----------------------------------------------- | --------- | -------- | ---------- | ------------- | -------- | -------------- | ---------- |
| [Intel/neural-chat-7b-v3-3]                     | ✓         | ✓        | ✓          | ✓             | ✓        | ✓              | -          |
| [meta-llama/Llama-2-7b-chat-hf]                 | ✓         | ✓        | ✓          | -             | ✓        | ✓              | ✓          |
| [meta-llama/Llama-2-70b-chat-hf]                | ✓         | -        | ✓          | -             | -        | ✓              | ✓          |
| [meta-llama/Meta-Llama-3-8B-Instruct]           | ✓         | ✓        | ✓          | -             | ✓        | ✓              | ✓          |
| [meta-llama/Meta-Llama-3-70B-Instruct]          | ✓         | -        | ✓          | -             | -        | ✓              | ✓          |
| [Phi-3]                                         | ✗         | Limit 4K | Limit 4K   | ✓             | Limit 4K | ✓              | -          |
| [Phi-4]                                         | ✗         | ✗        | ✗          | ✓             | ✗        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-8B]      | ✓         | -        | ✓          | ✓             | -        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B]     | ✓         | -        | ✓          | ✓             | -        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B]      | ✓         | -        | ✓          | ✓             | -        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B]      | ✓         | -        | ✓          | ✓             | -        | ✓              | -          |
| [mistralai/Mistral-Small-24B-Instruct-2501]     | ✓         | -        | ✓          | -             | -        | ✓              | -          |
| [mistralai/Mistral-Large-Instruct-2411]         | ✗         | -        | ✓          | -             | -        | ✓              | -          |
| [meta-llama/Llama-4-Scout-17B-16E-Instruct]     | -         | -        | -          | -             | -        | -              | ✓          |
| [meta-llama/Llama-4-Maverick-17B-128E-Instruct] | -         | -        | -          | -             | -        | -              | ✓          |
| [Qwen3-8B/14B/32B]                              | -         | -        | -          | ✓             | -        | -              | -          |

> **Note:** More details about supported models for vLLM-IPEX-XPU can be found at [supported-models](https://github.com/intel/llm-scaler/tree/main/vllm#3-supported-models).

### System Requirements for LLM Models

| Model                                     | Minimum Number of Gaudi Cards |
| ----------------------------------------- | ----------------------------- |
| Intel/neural-chat-7b-v3-3                 | 1                             |
| meta-llama/Llama-2-7b-chat-hf             | 1                             |
| meta-llama/Llama-2-70b-chat-hf            | 2                             |
| meta-llama/Meta-Llama-3-8B-Instruct       | 1                             |
| meta-llama/Meta-Llama-3-70B-Instruct      | 2                             |
| Phi-3                                     | -                             |
| Phi-4                                     | -                             |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B  | 1                             |
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 8                             |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B  | 2                             |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  | 4                             |
| mistralai/Mistral-Small-24B-Instruct-2501 | 1                             |
| mistralai/Mistral-Large-Instruct-2411     | 4                             |

> **Note:** Detailed hardware requirements will be provided soon.

---

## Start Microservice

### Support integrations

In this microservices, we have supported following backend LLM service as integrations, we will include TGI/vLLM/Ollama in this readme, for others, please refer to corresponding readmes.

- TGI
- VLLM
- Ollama
- [Bedrock](./README_bedrock.md)
- [Native](./README_native.md), based on optimum habana
- [Predictionguard](./README_predictionguard.md)
- [VLLM-IPEX](./README_vllm_ipex.md), based on B60 Graphics

### Clone OPEA GenAIComps

### Clone Repository

```bash
git clone https://github.com/opea-project/GenAIComps.git

export OPEA_GENAICOMPS_ROOT=$(pwd)/GenAIComps
```

### Prerequisites

- Obtain access to HuggingFace models and tokens:
  - [Create HuggingFace Account](https://huggingface.co/)
  - Set `HF_TOKEN` and `LLM_MODEL` as environment variables.

### Build Docker Image

#### Backend LLM Image

- For vLLM, refer to the [vLLM Build Guide](../../../third_parties/vllm/) to build the Docker images first.

- TGI and Ollama are not needed.

#### Build TextGen Microservice Image

```bash
cd ${OPEA_GENAICOMPS_ROOT}

docker build \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/llm-textgen:latest \
  -f comps/llms/src/text-generation/Dockerfile .
```

### Run Docker Service

You can start the service using either the CLI or Docker Compose. The `compose_text-generation.yaml` file will automatically start both endpoint and the microservice docker.

#### Setup Environment Variables

In order to start services, you need to setup the following environment variables first.

```bash
export LLM_ENDPOINT_PORT=8008
export TEXTGEN_PORT=9000
export host_ip=${host_ip}
export HF_TOKEN=${HF_TOKEN}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
```

#### Option A: Run Docker with CLI

1: Start backend LLM service ([TGI](../../../third_parties/tgi/), [vLLM](../../../third_parties/vllm/), [Ollama](../../../third_parties/ollama/)).

2: Start TextGen Microservice:

```bash
export LLM_COMPONENT_NAME="OpeaTextGenService"
docker run \
  --name="llm-textgen-server" \
  -p $TEXTGEN_PORT:9000 \
  --ipc=host \
  -e http_proxy=$http_proxy \
  -e https_proxy=$https_proxy \
  -e no_proxy=${no_proxy} \
  -e LLM_ENDPOINT=$LLM_ENDPOINT \
  -e HF_TOKEN=$HF_TOKEN \
  -e LLM_MODEL_ID=$LLM_MODEL_ID \
  -e LLM_COMPONENT_NAME=$LLM_COMPONENT_NAME \
  opea/llm-textgen:latest
```

#### Option B: Run with Docker Compose

Set `service_name` to match backend service.

```bash
export service_name="textgen-service-tgi"
# export service_name="textgen-service-tgi-gaudi"
# export service_name="textgen-service-vllm"
# export service_name="textgen-service-vllm-gaudi"
# export service_name="textgen-service-ollama"

cd ../../deployment/docker_compose/
docker compose -f compose_text-generation.yaml up ${service_name} -d
```

---

## Consume Microservice

### Check Service Status

```bash
curl http://${host_ip}:${TEXTGEN_PORT}/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### Verify microservice

You can set the following model parameters according to your actual needs, such as `max_tokens`, `stream`.

The `stream` parameter determines the format of the data returned by the API. It will return text string with `stream=false`, return text stream flow with `stream=true`.

```bash
# stream mode
curl http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
    -X POST \
    -d '{"model": "${LLM_MODEL_ID}", "messages": "What is Deep Learning?", "max_tokens":17}' \
    -H 'Content-Type: application/json'

curl http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
    -X POST \
    -d '{"model": "${LLM_MODEL_ID}", "messages": [{"role": "user", "content": "What is Deep Learning?"}], "max_tokens":17}' \
    -H 'Content-Type: application/json'

#Non-stream mode
curl http://${host_ip}:${TEXTGEN_PORT}/v1/chat/completions \
    -X POST \
    -d '{"model": "${LLM_MODEL_ID}", "messages": "What is Deep Learning?", "max_tokens":17, "stream":false}' \
    -H 'Content-Type: application/json'
```

<!--Below are links used in these document. They are not rendered: -->

[Intel/neural-chat-7b-v3-3]: https://huggingface.co/Intel/neural-chat-7b-v3-3
[meta-llama/Llama-2-7b-chat-hf]: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
[meta-llama/Llama-2-70b-chat-hf]: https://huggingface.co/meta-llama/Llama-2-70b-chat-hf
[meta-llama/Meta-Llama-3-8B-Instruct]: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
[meta-llama/Meta-Llama-3-70B-Instruct]: https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct
[Phi-3]: https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3
[Phi-4]: https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4
[HuggingFace]: https://huggingface.co/
[deepseek-ai/DeepSeek-R1-Distill-Llama-8B]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
[deepseek-ai/DeepSeek-R1-Distill-Llama-70B]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
[deepseek-ai/DeepSeek-R1-Distill-Qwen-32B]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
[deepseek-ai/DeepSeek-R1-Distill-Qwen-14B]: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
[mistralai/Mistral-Small-24B-Instruct-2501]: https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501
[mistralai/Mistral-Large-Instruct-2411]: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411
