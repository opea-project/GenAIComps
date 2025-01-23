# Reranking Microservice via TEI

`Text Embeddings Inference (TEI)` is a comprehensive toolkit designed for efficient deployment and serving of open source text embeddings models.
It enable us to host our own reranker endpoint seamlessly.

This README provides set-up instructions and comprehensive details regarding the reranking microservice via TEI.

---

## 🚀1. Start Microservice with Python (Option 1)

To start the Reranking microservice, you must first install the required python packages.

### 1.1 Install Requirements

```bash
pip install -r requirements.txt
```

### 1.2 Start TEI Service

```bash
export HF_TOKEN=${your_hf_api_token}
export RERANK_MODEL_ID="BAAI/bge-reranker-base"
export volume=$PWD/data

docker run -d -p 6060:80 -v $volume:/data -e http_proxy=$http_proxy -e https_proxy=$https_proxy --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $RERANK_MODEL_ID --hf-api-token $HF_TOKEN
```

### 1.3 Verify the TEI Service

```bash
export ip_address=$(hostname -I | awk '{print $1}')
curl ip_address:6060/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

### 1.4 Start Reranking Service with Python Script

```bash
export TEI_RERANKING_ENDPOINT="http://${ip_address}:6060"
export RERANK_TYPE=tei

cd GenAIComps/comps/rerankings/src
python opea_reranking_microservice.py
```

---

## 🚀2. Start Microservice with Docker (Option 2)

If you start an Reranking microservice with docker, the `rerank_tei.yaml` file will automatically start a TEI service with docker.

### 2.1 Setup Environment Variables

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export HF_TOKEN=${your_hf_api_token}
export TEI_RERANKING_ENDPOINT="http://${ip_address}:8808"
```

### 2.2 Build Docker Image

```bash
cd GenAIComps
docker build -t opea/reranking-tei:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/rerankings/src/Dockerfile .
```

To start a docker container, you have two options:

- A. Run Docker with CLI
- B. Run Docker with Docker Compose

You can choose one as needed.

### 2.3 Run Docker with CLI (Option A)

```bash
docker run -d --name="reranking-tei-server" -p 8000:8000 --ipc=host -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e TEI_RERANKING_ENDPOINT=$TEI_RERANKING_ENDPOINT -e HF_TOKEN=$HF_TOKEN RERANK_TYPE=tei opea/reranking:latest
```

### 2.4 Run Docker with Docker Compose (Option B)

```bash
docker compose -f comps/rerankings/deployment/docker_compose/rerank_tei.yaml up -d
```

---

## ✅3. Invoke Reranking Microservice

The Reranking microservice exposes following API endpoints:

- Check Service Status

  ```bash
  curl http://localhost:8000/v1/health_check \
    -X GET \
    -H 'Content-Type: application/json'
  ```

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
