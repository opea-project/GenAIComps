# Text To SQL  Microservice

## 1. Overview

## 🚀2. Start TextToSQL Microservice

```

### 2.1 Option 2. Start Microservice with Docker

#### 2.1.1 Build Microservices

```bash
cd GenAIComps/ # back to GenAIComps/ folder
docker build -t opea/comps-texttosql:latest -f comps/texttosql/langchain/Dockerfile .
```

#### 2.2.2 Start microservices

```bash
export ip_address=$(hostname -I | awk '{print $1}')
export LLM_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
export HF_TOKEN=${HUGGINGFACEHUB_API_TOKEN}

Run Directly

docker run  --runtime=runc --name="comps-langchain-texttosql"  -p 9090:8080 --ipc=host -e llm_endpoint_url=http://${ip_address}:8080 opea/comps-texttosql:latest


Run via docker compose.

docker compose -f docker_compose_texttosql.yaml up 

