# Embedding Microservice with OpenVINO Model Server (OVMS)

The **OVMS Embedding Microservice** is Intel’s highly optimized serving solution for generating embeddings using the OpenVINO Runtime. It efficiently converts text into high-dimensional vector embeddings with super fast inference on CPU.

---

## Table of Contents

1. [Start Microservice with `docker run`](#start-microservice-with-docker-run)
2. [Start Microservice with Docker Compose](#start-microservice-with-docker-compose)
3. [Consume Embedding Service](#consume-embedding-service)
4. [Tips for Better Understanding](#tips-for-better-understanding)

---

## Start Microservice with `docker run`

### Prepare Model and Export

Install requirements and export the model from HuggingFace Hub to local repository, convert to IR format and optionally quantize for faster startup:

```bash
pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
mkdir models
python export_model.py embeddings --source_model BAAI/bge-large-en-v1.5 --weight-format int8 --config_file_path models/config_embeddings.json --model_repository_path models --target_device CPU
```

### Run OVMS Docker Container

Run OVMS service container with model volume mounted and port mapping:

```bash
your_port=8090
docker run -p $your_port:8000 -v ./models:/models --name ovms-embedding-serving \
openvino/model_server:2025.0 --port 8000 --config_path /models/config_embeddings.json
```

### Test OVMS Service

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

### Build and Run Embedding Microservice Docker Image

1. Build the Docker image for the embedding microservice:

   ```bash
   cd ../../../
   docker build -t opea/embedding:latest \
   --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy \
   -f comps/embeddings/src/Dockerfile .
   ```

2. Run the embedding microservice connected to OVMS service:

   ```bash
   docker run -d --name="embedding-ovms-server" \
   -p 6000:6000 \
   --ipc=host \
   -e OVMS_EMBEDDING_ENDPOINT=$OVMS_EMBEDDING_ENDPOINT \
   -e MODEL_ID=$MODEL_ID \
   -e EMBEDDING_COMPONENT_NAME="OPEA_OVMS_EMBEDDING" \
   opea/embedding:latest
   ```

---

## Start Microservice with Docker Compose

Deploy both the OVMS Embedding Service and the Embedding Microservice using Docker Compose.

1. Export environment variables:

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

---

## Consume Embedding Service

### Check Service Status

Verify the embedding service is running:

```bash
curl http://localhost:6000/v1/health_check \
-X GET \
-H 'Content-Type: application/json'
```

### Use the Embedding Service API

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

---

## Tips for Better Understanding

1. **Port Mapping**  
   Ensure the ports are correctly mapped to avoid conflicts with other services.

2. **Model Selection**  
   Choose a model appropriate for your use case, like `"BAAI/bge-large-en-v1.5"` or `"BAAI/bge-base-en-v1.5"`.  
   The model should be exported into the model repository and set in the `MODEL_ID` environment variable when deploying the embedding wrapper service.

3. **Models repository Volume**  
   The `-v ./models:/models` flag ensures the model directory is correctly mounted into the container.

4. **Configuration JSON Selection**  
   The model repository can host multiple models. Select which models to serve by providing the correct configuration JSON file, such as `config_embeddings.json`.

5. **Kubernetes Deployment**  
   When deploying with Kubernetes, upload the model repository and configuration file to a persistent volume claim (PVC).  
   These will be mounted into the OVMS containers via [Helm chart](../../third_parties/ovms/deployment/kubernetes/README.md).

6. **Learn More about OVMS Embeddings API**  
   Refer to the [OVMS Embeddings API Documentation](https://docs.openvino.ai/2025/model-server/ovms_docs_rest_api_embeddings.html) for detailed API behavior.
