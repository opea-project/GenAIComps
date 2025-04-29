# LLM text generation Microservice

This microservice, designed for Language Model Inference (LLM), processes input consisting of a query string and associated reranked documents. It constructs a prompt based on the query and documents, which is then used to perform inference with a large language model. The service delivers the inference results as output.

A prerequisite for using this microservice is that users must have a LLM text generation service (etc., TGI, vLLM) already running. Users need to set the LLM service's endpoint into an environment variable. The microservice utilizes this endpoint to create an LLM object, enabling it to communicate with the LLM service for executing language model operations.

Overall, this microservice offers a streamlined way to integrate large language model inference into applications, requiring minimal setup from the user beyond initiating a TGI/vLLM service and configuring the necessary environment variables. This allows for the seamless processing of queries and documents to generate intelligent, context-aware responses.

## Validated LLM Models

| Model                                                                                                                 | TGI-Gaudi | vLLM-CPU | vLLM-Gaudi | OVMS     | Optimum-Habana | SGLANG-CPU |
| --------------------------------------------------------------------------------------------------------------------- | --------- | -------- | ---------- | -------- | -------------- | ---------- |
| [Intel/neural-chat-7b-v3-3]                                                                                           | ✓         | ✓        | ✓          | ✓        | ✓              | -          |
| [meta-llama/Llama-2-7b-chat-hf]                                                                                       | ✓         | ✓        | ✓          | ✓        | ✓              | ✓          |
| [meta-llama/Llama-2-70b-chat-hf]                                                                                      | ✓         | -        | ✓          | -        | ✓              | ✓          |
| [meta-llama/Meta-Llama-3-8B-Instruct]                                                                                 | ✓         | ✓        | ✓          | ✓        | ✓              | ✓          |
| [meta-llama/Meta-Llama-3-70B-Instruct]                                                                                | ✓         | -        | ✓          | -        | ✓              | ✓          |
| [Phi-3]                                                                                                               | x         | Limit 4K | Limit 4K   | Limit 4K | ✓              | -          |
| [Phi-4]                                                                                                               | x         | x        | x          | x        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-8B]                                                                            | ✓         | -        | ✓          | -        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B]                                                                           | ✓         | -        | ✓          | -        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B]                                                                            | ✓         | -        | ✓          | -        | ✓              | -          |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B]                                                                            | ✓         | -        | ✓          | -        | ✓              | -          |
| [mistralai/Mistral-Small-24B-Instruct-2501]                                                                           | ✓         | -        | ✓          | -        | ✓              | -          |
| [mistralai/Mistral-Large-Instruct-2411]                                                                               | x         | -        | ✓          | -        | ✓              | -          |
| [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)         | -         | -        | -          | -        | -              | ✓          |
| [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | -         | -        | -          | -        | -              | ✓          |

### System Requirements for LLM Models

| Model                                       | Minimum number of Gaudi cards |
| ------------------------------------------- | ----------------------------- |
| [Intel/neural-chat-7b-v3-3]                 | 1                             |
| [meta-llama/Llama-2-7b-chat-hf]             | 1                             |
| [meta-llama/Llama-2-70b-chat-hf]            | 2                             |
| [meta-llama/Meta-Llama-3-8B-Instruct]       | 1                             |
| [meta-llama/Meta-Llama-3-70B-Instruct]      | 2                             |
| [Phi-3]                                     | x                             |
| [Phi-4]                                     | x                             |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-8B]  | 1                             |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B] | 8                             |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B]  | 2                             |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B]  | 4                             |
| [mistralai/Mistral-Small-24B-Instruct-2501] | 1                             |
| [mistralai/Mistral-Large-Instruct-2411]     | 4                             |

> NOTE: Detailed system requirements coming soon.

## Support integrations

In this microservices, we have supported following backend LLM service as integrations, we will include TGI/vLLM/Ollama in this readme, for others, please refer to corresponding readmes.

- TGI
- VLLM
- Ollama
- [Bedrock](./README_bedrock.md)
- [Native](./README_native.md), based on optimum habana
- [Predictionguard](./README_predictionguard.md)

## Clone OPEA GenAIComps

Clone this repository at your desired location and set an environment variable for easy setup and usage throughout the instructions.

```bash
git clone https://github.com/opea-project/GenAIComps.git

export OPEA_GENAICOMPS_ROOT=$(pwd)/GenAIComps
```

## Prerequisites

For TGI/vLLM, You must create a user account with [HuggingFace] and obtain permission to use the gated LLM models by adhering to the guidelines provided on the respective model's webpage. The environment variables `LLM_MODEL` would be the HuggingFace model id and the `HF_TOKEN` is your HuggugFace account's "User Access Token".

## 🚀Start Microservice with Docker

In order to start the microservices with docker, you need to build the docker images first for the microservice.

### 1. Build Docker Image

#### 1.1 Prepare backend LLM docker image.

If you want to use vLLM backend, refer to [vLLM](../../../third_parties/vllm/) to build vLLM docker images first.

No need for TGI or Ollama.

#### 1.2 Prepare TextGen docker image.

```bash
# Build the microservice docker
cd ${OPEA_GENAICOMPS_ROOT}

docker build \
  --build-arg https_proxy=$https_proxy \
  --build-arg http_proxy=$http_proxy \
  -t opea/llm-textgen:latest \
  -f comps/llms/src/text-generation/Dockerfile .
```

### 2. Start LLM Service with the built image

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed. If you start an LLM microservice with docker compose, the `compose_text-generation.yaml` file will automatically start both endpoint and the microservice docker.

#### 2.1 Setup Environment Variables

In order to start services, you need to setup the following environment variables first.

```bash
export LLM_ENDPOINT_PORT=8008
export TEXTGEN_PORT=9000
export host_ip=${host_ip}
export HF_TOKEN=${HF_TOKEN}
export LLM_ENDPOINT="http://${host_ip}:${LLM_ENDPOINT_PORT}"
export LLM_MODEL_ID="Intel/neural-chat-7b-v3-3"
```

#### 2.2 Run Docker with CLI (Option A)

Step 1: Start the backend LLM service

Please refer to [TGI](../../../third_parties/tgi/), [vLLM](../../../third_parties/vllm/), [Ollama](../../../third_parties/ollama/) guideline to start a backend LLM service.

Step 2: Start the TextGen microservices

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

#### 2.3 Run Docker with Docker Compose (Option B)

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

## 🚀3. Consume LLM Service

### 3.1 Check Service Status

```bash
curl http://${host_ip}:${TEXTGEN_PORT}/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 3.1 Verify microservice

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
