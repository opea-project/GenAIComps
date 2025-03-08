# LLM OVMS Microservice

LLM OVMS microservice uses [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server). It can efficient generate text on Intel CPU using a set of optimizations technics list continuous batching, paged attention, prefix caching, speculative decoding and many other.


## 🚀1. Start OVMS Microservice

### 1.1 Prepare models

In order to start LLM OVMS service, you need to export the models from Hugging Faces Hub to IR format. At the some time model will be converted to IR format and optionally quantized. It speedup starting the service and avoids copying the model from Internet each time the container starts.

    ```
    pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
    mkdir models
    python export_model.py text-generation --source_model Qwen/Qwen2-7B-Instruct --weight-format int8 --config_file_path models/config_llm.json --model_repository_path models --target_device CPU
    ```

Change the `source_model` as needed.

### 1.2 **Start the OVMS container**:
   Replace `your_port` with desired values to start the service.

   ```bash
   your_port=8090
   docker run -p $your_port:8000 -v ./models:/models --name ovms-llm-serving \
   openvino/model_server:2025.0 --port 8000 --config_path /models/config_llm.json
   ```

### 1.3  **Test the OVMS container**:
   OVMS exposes REST API compatible with OpenAI API. Both `completions` and `chat/completions` are supported.
   Run the following command to check if the service is up and running.

   ```bash
    curl -s http://localhost:8090/v3/chat/completions   \
    -H "Content-Type: application/json"   \
    -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "max_tokens":30, "temperature":0,
    "stream":false,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What are the 3 main tourist attractions in Paris?"
      }
    ]
    }'
   ```


## 🚀2. Starting OPEA LLM microservice

### 2.1  Building the image

```bash
cd ../../../../../
docker build -t opea/llm-textgen-ovms:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/llms/src/text-generation/Dockerfile .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 2.2a Run Docker with CLI (Option A)

```bash
export OVMS_LLM_ENDPOINT=http://localhost:8090
export MODEL_ID=Qwen/Qwen2-7B-Instruct
docker run -d --name="llm-ovms-server" -p 9000:9000  \
-e MODEL_ID=${MODEL_ID} \
-e LLM_COMPONENT_NAME=OpeaTextGenOVMS \
-e OVMS_LLM_ENDPOINT=${OVMS_LLM_ENDPOINT} \
opea/llm-textgen-ovms:latest
```

### 2.2b Run Docker with Docker Compose (Option B)

```bash
export service_name="textgen-ovms"
cd comps/llms/deployment/docker_compose
docker compose -f compose_text-generation.yaml up ${service_name} -d
```

## 🚀2. Consume LLM Service

### 2.1 Check Service Status

```bash
curl http://localhost:9000/v1/health_check\
  -X GET \
  -H 'Content-Type: application/json'
```

### 2.2 Consume LLM Service

```bash
curl http://localhost:9000/v1/chat/completions\
  -X POST \
  -d '{"messages":"What is Deep Learning?"}' \
  -H 'Content-Type: application/json'
```

```bash
curl http://localhost:9000/v1/chat/completions\
  -X POST \
  -d '{"messages":"What is Deep Learning?", "stream": true}' \
  -H 'Content-Type: application/json'
```