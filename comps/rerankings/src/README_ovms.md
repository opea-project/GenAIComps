# 🌟 Reranking Microservice with OVMS

This guide walks you through starting, deploying, and consuming the **OVMS Reranking Microservice**. 🚀
It is Intel highly optimized serving solution which employs OpenVINO Runtime for super fast inference on CPU.

---

## 📦 1. Prepare the model in the model repository

This step will export the model from HuggingFace Hub to the local models repository. At the some time model will be converted to IR format and optionally quantized.  
It speedup starting the service and avoids copying the model from Internet each time the container starts.

    ```
    pip3 install -r https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/requirements.txt
    curl https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py -o export_model.py
    mkdir models
    python export_model.py rerank --source_model BAAI/bge-reranker-large --weight-format int8 --config_file_path models/config_reranking.json --model_repository_path models --target_device CPU
    ```

## 📦 2. Start Microservice with Docker

### 🔹 2.1 Start Reranking Service with OVMS

1. **Start the OVMS service**:

- For Xeon CPU:

  ```bash
  your_port=8001
  docker run -p $your_port:8001 -v ./models:/models --name ovms-rerank-serving \
  openvino/model_server:2025.0 --port 8001 --config_path /models/config_reranking.json
  ```

2. **Verify the OVMS Service**:

   Run the following command to check if the service is up and running.

   ```bash
     curl http://localhost:$your_port/v3/rerank \
         -X POST \
         -H 'Content-Type: application/json' \
         -d '{ "model": "BAAI/bge-reranker-large", "query": "welcome", "documents":["Deep Learning is not...", "Deep learning is..."]}'
   ```

### 🔹 1.2 Build Docker Image and Run Docker with CLI

1. Build the Docker image for the reranking microservice:

   ```bash
   docker build -t opea/reranking:comps --build-arg SERVICE=ovms -f comps/rerankings/src/Dockerfile .
   ```

2. Run the reranking microservice and connect it to the OVMS service:

   ```bash
    export OVMS_RERANKING_PORT=8001
    export OVMS_RERANKING_ENDPOINT="http://localhost:${OVMS_RERANKING_PORT}"
    export MODEL_ID=BAAI/bge-reranker-large

   docker run -d --rm --name="reranking-ovms-server" -e LOGFLAG=True  -p 8000:8000 --ipc=host -e OVMS_RERANKING_ENDPOINT=$OVMS_RERANKING_ENDPOINT -e RERANK_COMPONENT_NAME="OPEA_OVMS_RERANKING" -e MODEL_ID=$MODEL_ID opea/reranking:comps
   ```

## 📦 3. Start Microservice with docker compose

Deploy both the OVMS Reranking Service and the Reranking Microservice using Docker Compose.

🔹 Steps:

1. Set environment variables:

   ```bash
    export MODEL_ID="BAAI/bge-reranker-large"
    export OVMS_RERANKING_PORT=12005
    export RERANK_PORT=8000
    export host_ip=$(hostname -I | awk '{print $1}')
    export OVMS_RERANKING_ENDPOINT="http://${host_ip}:${OVMS_RERANKING_PORT}"
    export TAG=comps

   ```

2. Navigate to the Docker Compose directory:

   ```bash
   cd comps/rerankings/deployment/docker_compose/
   ```

3. Start the services:

- For Xeon CPU:

  ```bash
   docker compose up ovms-reranking-server -d
  ```

## 📦 4. Consume Reranking Service

### 🔹 4.1 Check Service Status

- Verify the reranking service is running:

  ```bash
    curl http://localhost:8000/v1/health_check \
    -X GET \
    -H 'Content-Type: application/json'
  ```

### 🔹 4.2 Use the Reranking Service API

- Execute reranking process by providing query and documents

  ```bash
  curl http://localhost:8000/v1/reranking -X POST -H 'Content-Type: application/json' \
    -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}'
  ```

  - You can add the parameter `top_n` to specify the return number of the reranker model, default value is 1.

  ```bash
  curl http://localhost:8000/v1/reranking \
    -X POST \
    -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}], "top_n":2}' \
    -H 'Content-Type: application/json'
  ```

## ✨ Tips for Better Understanding:

1. Port Mapping:
   Ensure the ports are correctly mapped to avoid conflicts with other services.

2. Model Selection:
   Choose a model appropriate for your use case, like "BAAI/bge-reranker-large" or "BAAI/bge-reranker-v2-m3".
   It should be exported to the models repository and set in 'MODEL_ID' env in the deployment of the OPEA API wrapper.

3. Models repository Volume:
   The `-v ./models:/models` flag ensures the models directory is correctly mounted.

4. Select correct configuration JSON file
   Models repository can host multiple models. Choose the models to be served by selecting the right configuration file.
   In the example above `config_reranking.json`

5. Upload the models to persistent volume claim in Kubernetes
   Models repository with configuration JSON file will be mounted in the OVMS containers when deployed via [helm chart](../../third_parties/ovms/deployment/kubernetes/README.md).

6. Learn more about [OVMS rerank API](https://docs.openvino.ai/2025/model-server/ovms_docs_rest_api_rerank.html)
