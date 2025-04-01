# 🌟 Reranking Microservice with TEI

`Text Embeddings Inference (TEI)` is a comprehensive toolkit designed for efficient deployment and serving of open source text embeddings models.
It enables us to host our own reranker endpoint seamlessly.

This README provides set-up instructions and comprehensive details regarding the reranking microservice via TEI.

---

## 📦 1. Start Microservice with Docker

### 🔹 1.1 Start Reranking Service with TEI

1. **Start the TEI service**:

- For Gaudi HPU:

  ```bash
    export HF_TOKEN=${your_hf_api_token}
    export RERANK_MODEL_ID="BAAI/bge-reranker-base"
    export volume=$PWD/data

    docker run -p 12005:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e MAX_WARMUP_SEQUENCE_LENGTH=512 --cap-add=sys_nice --ipc=host ghcr.io/huggingface/tei-gaudi:latest --model-id $RERANK_MODEL_ID --hf-api-token $HF_TOKEN --auto-truncate

  ```

- For Xeon CPU:

  ```bash
    export HF_TOKEN=${your_hf_api_token}
    export RERANK_MODEL_ID="BAAI/bge-reranker-base"
    export volume=$PWD/data

    docker run -d -p 12005:80 -v $volume:/data -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 --model-id $RERANK_MODEL_ID --hf-api-token $HF_TOKEN --auto-truncate
  ```

2. **Verify the TEI Service**:

   Run the following command to check if the service is up and running.

   ```bash
     export host_ip=$(hostname -I | awk '{print $1}')
     curl $host_ip:12005/rerank \
         -X POST \
         -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
         -H 'Content-Type: application/json'
   ```

### 🔹 1.2 Build Docker Image and Run Docker with CLI

1. Build the Docker image for the reranking microservice:

   ```bash
    docker build --no-cache \
      -t opea/reranking:comps \
      --build-arg https_proxy=$https_proxy \
      --build-arg http_proxy=$http_proxy \
      --build-arg SERVICE=tei \
      -f comps/rerankings/src/Dockerfile .
   ```

2. Run the reranking microservice and connect it to the TEI service:

   ```bash
    export TEI_RERANKING_PORT=12005
    export HF_TOKEN=${your_hf_api_token}
    export host_ip=$(hostname -I | awk '{print $1}')
    export TEI_RERANKING_ENDPOINT="http://${host_ip}:${TEI_RERANKING_PORT}"

   docker run -d --name="reranking-tei-server" -e LOGFLAG=True  -p 8000:8000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TEI_RERANKING_ENDPOINT=$TEI_RERANKING_ENDPOINT -e HF_TOKEN=$HF_TOKEN  -e RERANK_COMPONENT_NAME="OPEA_TEI_RERANKING"  opea/reranking:comps
   ```

## 📦 2. Start Microservice with docker compose

Deploy both the TEI Reranking Service and the Reranking Microservice using Docker Compose.

🔹 Steps:

1. Set environment variables:

   ```bash
    export RERANK_MODEL_ID="BAAI/bge-reranker-base"
    export TEI_RERANKING_PORT=12005
    export RERANK_PORT=8000
    export host_ip=$(hostname -I | awk '{print $1}')
    export TEI_RERANKING_ENDPOINT="http://${host_ip}:${TEI_RERANKING_PORT}"
    export TAG=comps

   ```

2. Navigate to the Docker Compose directory:

   ```bash
   cd comps/rerankings/deployment/docker_compose/
   ```

3. Start the services:

- For Gaudi HPU:

  ```bash
   docker compose up reranking-tei-gaudi -d
  ```

- For Xeon CPU:

  ```bash
   docker compose up reranking-tei -d
  ```

## 📦 3. Consume Reranking Service

### 🔹 3.1 Check Service Status

- Verify the reranking service is running:

  ```bash
    curl http://localhost:8000/v1/health_check \
    -X GET \
    -H 'Content-Type: application/json'
  ```

### 🔹 3.2 Use the Reranking Service API

- Execute reranking process by providing query and documents

  ```bash
  curl http://localhost:8000/v1/reranking \
    -X POST \
    -d '{"initial_query":"What is Deep Learning?", "retrieved_docs": [{"text":"Deep Learning is not..."}, {"text":"Deep learning is..."}]}' \
    -H 'Content-Type: application/json'
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
   Choose a model appropriate for your use case, like "BAAI/bge-reranker-base".

3. Environment Variables:
   Use http_proxy and https_proxy for proxy setup if necessary.

4. Data Volume:
   The `-v ./data:/data` flag ensures the data directory is correctly mounted.
