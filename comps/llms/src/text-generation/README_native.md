# LLM Native Microservice

LLM Native microservice uses [optimum-habana](https://github.com/huggingface/optimum-habana) for model initialization and warm-up, focusing solely on large language models (LLMs). It operates without frameworks like TGI/VLLM, using PyTorch directly for inference, and supports only non-stream formats. This streamlined approach optimizes performance on Habana hardware.

## 🚀1. Start Microservice

### 1.1 Setup Environment Variables

In order to start Native LLM service, you need to setup the following environment variables first.

For LLM model, both `Qwen` and `Falcon3` models are supported. Users can set different models by changing the `LLM_MODEL_ID` below.

```bash
export LLM_MODEL_ID="Qwen/Qwen2-7B-Instruct"
export HF_TOKEN="your_huggingface_token"
export TEXTGEN_PORT=10512
export host_ip=${host_ip}
```

### 1.2 Build Docker Image

```bash
cd ../../../../../
docker build -t opea/llm-textgen-gaudi:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile.intel_hpu .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 1.3 Run Docker with CLI (Option A)

```bash
docker run -d --runtime=habana --name="llm-native-server" -p 9000:9000 -e https_proxy=$https_proxy -e http_proxy=$http_proxy -e TOKENIZERS_PARALLELISM=false -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -e LLM_MODEL_ID=${LLM_MODEL_ID} opea/llm-textgen-gaudi:latest
```

### 1.4 Run Docker with Docker Compose (Option B)

```bash
export service_name="textgen-native-gaudi"
cd comps/llms/deployment/docker_compose
docker compose -f compose_text-generation.yaml up ${service_name} -d
```

## 🚀2. Consume LLM Service

### 2.1 Check Service Status

```bash
curl http://${your_ip}:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 2.2 Consume LLM Service

```bash
curl http://${your_ip}:9000/v1/chat/completions\
  -X POST \
  -d '{"messages":"What is Deep Learning?"}' \
  -H 'Content-Type: application/json'
```
