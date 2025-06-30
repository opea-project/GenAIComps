# 🌟 Reranking Microservice with VideoQnA

This README provides set-up instructions and comprehensive details regarding the reranking microservice with VideoQnA.
This microservice is designed that do result rerank for VideoQnA use case. Local rerank is used rather than rerank model.

For the `VideoQnA` usecase, during the data preparation phase, frames are extracted from videos and stored in a vector database.
To identify the most relevant video, we count the occurrences of each video source among the retrieved data with rerank function `get_top_doc`.
This sorts the video as a descending list of names, ranked by their degree of match with the query.
Then we could send the `top_n` videos to the downstream LVM.

---

## 📦 1. Start Microservice with Docker

### 🔹 1.1 Build Docker Image and Run Docker with CLI

1. Build the Docker image for the reranking microservice:

   ```bash
      docker build --no-cache \
      -t opea/reranking:comps \
      --build-arg https_proxy=$https_proxy \
      --build-arg http_proxy=$http_proxy \
      --build-arg SERVICE=svideoqna \
      -f comps/rerankings/src/Dockerfile .
   ```

2. Run the reranking microservice and connect it to the VideoQnA service:

   ```bash
    docker run -d --name "reranking-videoqna-server" \
      -p 8000:8000 \
      --ipc=host \
      -e no_proxy=${no_proxy} \
      -e http_proxy=${http_proxy} \
      -e https_proxy=${https_proxy} \
      -e CHUNK_DURATION=${CHUNK_DURATION} \
      -e RERANK_COMPONENT_NAME="OPEA_VIDEO_RERANKING" \
      -e FILE_SERVER_ENDPOINT=${FILE_SERVER_ENDPOINT} \
      opea/reranking:comps
   ```

## 📦 2. Start Microservice with docker compose

Deploy both the Videoqna Reranking Service and the Reranking Microservice using Docker Compose.

🔹 Steps:

1. Set environment variables:

   ```bash
    export TEI_RERANKING_PORT=12006
    export RERANK_PORT=8000
    export host_ip=$(hostname -I | awk '{print $1}')
    export TEI_RERANKING_ENDPOINT="http://${host_ip}:${TEI_RERANKING_PORT}"
    export TAG=comps
   ```

2. Navigate to the Docker Compose directory:

   ```bash
   cd comps/rerankings/deployment/docker_compose
   ```

3. Start the services:

   ```bash
    docker compose up reranking-videoqna -d
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

2. Environment Variables:

   - Use http_proxy and https_proxy for proxy setup if necessary.
   - CHUNK_DURATION: target chunk duration, should be aligned with VideoQnA dataprep. Default 10s.

3. Data Volume:
   The `-v ./data:/data` flag ensures the data directory is correctly mounted.
