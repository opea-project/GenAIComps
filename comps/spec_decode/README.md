# Speculative Decoding Microservice

This microservice, designed for Speculative Decoding, processes input consisting of a query string. It constructs a prompt based on the query and documents, which is then used to perform inference with a large language model. The service delivers the inference results as output.

A prerequisite for using this microservice is that users must have a LLM text generation service already running. Users need to set the LLM service's endpoint into an environment variable. The microservice utilizes this endpoint to create an speculative decoding object, enabling it to communicate with the speculative decoding service for executing language model operations.

Overall, this microservice offers a streamlined way to integrate large language model inference into applications, requiring minimal setup from the user beyond initiating a vLLM service and configuring the necessary environment variables. This allows for the seamless processing of queries and documents to generate intelligent, context-aware responses.

## 🚀1. Start Microservice with Python (Option 1)

To start the LLM microservice, you need to install python packages first.

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start Speculative Decoding Service

#### 1.2.1 Start vLLM Service

```bash
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
docker run -it --name vllm_service -p 8008:8008 -e HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN} -v ./data:/data opea/vllm:cpu /bin/bash -c "cd / && export VLLM_CPU_KVCACHE_SPACE=40 && python3 -m vllm.entrypoints.openai.api_server --model ${your_hf_llm_model} --speculative_model ${your_speculative_model} --num_speculative_tokens ${your_speculative_tokens} --use-v2-block-manager --tensor-parallel-size 1 --port 8008"
```

### 1.3 Verify the Speculative Decoding Service

#### 1.3.2 Verify the vLLM Service

```bash
curl http://${your_ip}:8008/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": ${your_hf_llm_model},
  "prompt": "What is Deep Learning?",
  "max_tokens": 32,
  "temperature": 0
  }'
```

### 1.4 Start Speculative Decoding Service with Python Script

#### 1.4.1 Start the vLLM Service

```bash
export vLLM_LLM_ENDPOINT="http://${your_ip}:8008"
python text-generation/vllm/llm.py
```

## 🚀2. Start Microservice with Docker (Option 2)

If you start an LLM microservice with docker, the `docker_compose_spec_decode.yaml` file will automatically start a vLLM service with docker.

### 2.1 Setup Environment Variables

In order to start vLLM and LLM services, you need to setup the following environment variables first.

```bash
export HUGGINGFACEHUB_API_TOKEN=${your_hf_api_token}
export vLLM_LLM_ENDPOINT="http://${your_ip}:8008"
export LLM_MODEL_ID=${your_hf_llm_model}
export SPEC_MODEL_ID=${your_hf_spec_model}
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="opea/spec_decode"
```

### 2.2 Build Docker Image

#### 2.2.1 vLLM

Build vllm docker.

```bash
bash build_docker_vllm.sh
```

Build microservice docker.

```bash
cd ../../../../
docker build -t opea/spec_decode-vllm:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/spec_decode/text-generation/vllm/docker/Dockerfile.microservice .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 2.3 Run Docker with CLI (Option A)

#### 2.3.1 vLLM

Start vllm endpoint.

```bash
bash launch_vllm_service.sh
```

Start vllm microservice.

```bash
docker run --name="llm-vllm-server" -p 9000:9000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=${no_proxy} -e vLLM_LLM_ENDPOINT=$vLLM_LLM_ENDPOINT -e HUGGINGFACEHUB_API_TOKEN=$HUGGINGFACEHUB_API_TOKEN -e SPEC_MODEL_ID=$SPEC_MODEL_ID -e LLM_MODEL_ID=$LLM_MODEL_ID opea/specs_decode-vllm:latest
```

### 2.4 Run Docker with Docker Compose (Option B)

#### 2.4.1 vLLM

```bash
cd text-generation/vllm
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

You can set the following model parameters according to your actual needs, such as `max_new_tokens`, `streaming`.

The `streaming` parameter determines the format of the data returned by the API. It will return text string with `streaming=false`, return text streaming flow with `streaming=true`.

```bash
# non-streaming mode
curl http://${your_ip}:9000/v1/spec_decode/completions \
  -X POST \
  -d '{"query":"What is Deep Learning?","max_new_tokens":17,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":false}' \
  -H 'Content-Type: application/json'

# streaming mode
curl http://${your_ip}:9000/v1/spec_decode/completions \
  -X POST \
  -d '{"query":"What is Deep Learning?","max_new_tokens":17,"top_k":10,"top_p":0.95,"typical_p":0.95,"temperature":0.01,"repetition_penalty":1.03,"streaming":true}' \
  -H 'Content-Type: application/json'
```
