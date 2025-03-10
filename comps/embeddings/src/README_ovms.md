# 🌟 Embedding Microservice with OpenVINO Model Server

This guide walks you through starting, deploying, and consuming the **OVMS Embeddings Microservice**. 🚀
It is Intel highly optimized serving solution which employs OpenVINO Runtime for super fast inference on CPU.

---

## 📦 1. Start Microservice with `docker run`

### 🔹 1.1 Start Embedding Service with OVMS

1. Prepare the model in the model repository
   This step will export the model from HuggingFace Hub to the local models repository. At the some time model will be converted to IR format and optionally quantized.  
   It speedup starting the service and avoids copying the model from Internet each time the container starts.

   ```
   pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
   curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
   mkdir models
   python export_model.py embeddings --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --config_file_path models/config_embeddings.json --model_repository_path models --target_device CPU
   ```

2. **Test the OVMS service**:
   Run the following command to check if the service is up and running.

```bash
your_port=8090
docker run -p $your_port:8000 -v ./models:/models --name ovms-embedding-serving \
openvino/model_server:2025.0 --port 8000 --config_path /models/config_embeddings.json
```

3. **Test the OVMS service**:
   Run the following command to check if the service is up and running.

   ```bash
   curl http://localhost:$your_port/v3/embeddings \
   -X POST \
   -H 'Content-Type: application/json'
   -d '{
   "model": "BAAI/bge-large-en-v1.5",
   "input":"What is Deep Learning?"
   }'
   ```

### 🔹 1.2 Build Docker Image and Run Docker with CLI

1. Build the Docker image for the embedding microservice:

   ```bash
   cd ../../../
   docker build -t opea/embedding:latest \
   --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy \
   -f comps/embeddings/src/Dockerfile .
   ```

2. Run the embedding microservice and connect it to the OVMS service:

   ```bash
   docker run -d --name="embedding-ovms-server" \
   -p 6000:6000 \
   --ipc=host \
   -e OVMS_EMBEDDING_ENDPOINT=$OVMS_EMBEDDING_ENDPOINT \
   -e MODEL_ID=$MODEL_ID \
   -e EMBEDDING_COMPONENT_NAME="OPEA_OVMS_EMBEDDING" \
   opea/embedding:latest
   ```

## 📦 2. Start Microservice with docker compose

Deploy both the OVMS Embedding Service and the Embedding Microservice using Docker Compose.

🔹 Steps:

1. Set environment variables:

   ```bash
   export host_ip=${your_ip_address}
   export MODEL_ID="BAAI/bge-large-en-v1.5"
   export OVMS_EMBEDDER_PORT=8090
   export EMBEDDER_PORT=6000
   export OVMS_EMBEDDING_ENDPOINT="http://${host_ip}:${OVMS_EMBEDDER_PORT}"
   ```

2. Navigate to the Docker Compose directory:

   ```bash
   cd comps/embeddings/deployment/docker_compose/
   ```

3. Start the services:

   ```bash
   docker compose up ovms-embedding-server -d
   ```

## 📦 3. Consume Embedding Service

### 🔹 3.1 Check Service Status

Verify the embedding service is running:

```bash
curl http://localhost:6000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### 🔹 3.2 Use the Embedding Service API

The API is compatible with the [OpenAI API](https://platform.openai.com/docs/api-reference/embeddings).

1. Single Text Input

   ```bash
   curl http://localhost:6000/v1/embeddings \
   -X POST \
   -d '{"input":"Hello, world!"}' \
   -H 'Content-Type: application/json'
   ```

2. Multiple Text Inputs with Parameters

   ```bash
   curl http://localhost:6000/v1/embeddings \
   -X POST \
   -d '{"input":["Hello, world!","How are you?"], "dimensions":100}' \
   -H 'Content-Type: application/json'
   ```

## ✨ Tips for Better Understanding:

1. Port Mapping:
   Ensure the ports are correctly mapped to avoid conflicts with other services.

2. Model Selection:
   Choose a model appropriate for your use case, like "BAAI/bge-large-en-v1.5" or "BAAI/bge-base-en-v1.5".
   It should be exported to the models repository and set in 'MODEL_ID' env in the deployment of the OPEA API wrapper.

3. Models repository Volume:
   The `-v ./models:/models` flag ensures the models directory is correctly mounted.

4. Select correct configuration JSON file
   Models repository can host multiple models. Choose the models to be served by selecting the right configuration file.
   In the example above `config_embeddings.json`

5. Upload the models to persistent volume claim in Kubernetes
   Models repository with configuration JSON file will be mounted in the OVMS containers when deployed via [helm chart](../../third_parties/ovms/deployment/kubernetes/README.md).

6. Learn more about [OVMS embeddings API](https://docs.openvino.ai/2025/model-server/ovms_docs_rest_api_embeddings.html)
